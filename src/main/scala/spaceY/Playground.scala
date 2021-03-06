package spaceY

import java.awt.Dimension
import java.io.File

import javax.swing.{JFrame, WindowConstants}
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import rx._
import spaceY.DoubleQTraining.CheckPoint
import spaceY.Geometry2D._
import spaceY.NetworkModel.ModelParams
import spaceY.Simulator.{FullSimulation, NoInfo, PolicyInfo, WorldBound}

import scala.util.Random

object Playground {

  val deltaT: Double = 1.0/25.0
  val world = World(gravity = Vec2.down*10, maxThrust = 30, deltaT = deltaT)

  val useGUI = true

  def main(args: Array[String]): Unit = {
    val tasks = (27 until 40).map{ i =>
      val seed = i
      val rand = new Random(seed)
      (0 until 5).foreach(_ => rand.nextInt())

      def select[A](xs: A*): A = {
        SimpleMath.randomSelect(rand)(xs.toIndexedSeq)
      }

      val params = TrainingParams(
        modelParams = ModelParams(
          outLayerActivation = select(Activation.RELU, Activation.SIGMOID),
          sizes = NetworkModel.getModelSizes(layers = select(3,4,5), baseNeurons = select(64,128)),
          updater = select(Updater.ADAM, Updater.NESTEROVS),
          learningRate = SimpleMath.expInterpolate(0.0005, 0.01,10)(rand.nextDouble())
        ),
        batchSize = select(64,128),
        batchesPerDataCollect = SimpleMath.expInterpolate(5, 50,10)(rand.nextDouble()).toInt,
        seed = seed,
        gamma = select(0.999,0.99,1.0),
        replayBufferSize = SimpleMath.expInterpolate(50*100, 50*1000, 10)(rand.nextDouble()).toInt,
        updateDataNum = select(10 to 80 :_*),
        copyInterval = select(25 to 100 :_*),
//        exploreAmount = SimpleMath.expInterpolate(0.005,0.05,10)(rand.nextDouble()),
//        exploreDecay = SimpleMath.expInterpolate(5000,50000,10)(rand.nextDouble())
      )

      i -> params
    }
//    SimpleMath.processMap(args, tasks, processNum = 10, mainClass = this){
    SimpleMath.parallelMap(tasks, threadNum = 2){
      case (id, params) =>
        train(params, ioId = id)
    }
  }

  def train(params: TrainingParams, ioId: Int): Unit = {

    val bound = WorldBound(width = 200, height = 150)
    val availableActions: IS[Action] = {
      for (
        rotate <- IS(1.0, 0.0, -1.0);
        thrust <- 0.0 to 1.0 by 1.0 / 6
      ) yield Action(rotate, thrust)
    }

    val initStates = IS(
      State(pos = Vec2(20, 50), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 20, fuelLeft = 10),
      State(pos = Vec2(20, 125), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 75, fuelLeft = 10),
      State(pos = Vec2(-75, 100), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 75, fuelLeft = 10),
      State(pos = Vec2(0, 125), velocity = Vec2.zero, rotation = Rotation2(-0.65), goalX = 25, fuelLeft = 10),
      State(pos = Vec2(-25, 75), velocity = Vec2(-10, -10), rotation = Rotation2(0), goalX = 25, fuelLeft = 10)
    )

    def sampleInitState(random: Random): State = {
      def between(from: Double, to: Double) = SimpleMath.linearInterpolate(from, to)(random.nextDouble())

      State(
        pos = Vec2(between(-80, 80), between(50, 125)),
        velocity = Vec2(between(-20, 20), between(-20, 20)),
        rotation = Rotation2(SimpleMath.cubic(between(-1.0, 1.0))),
        goalX = between(-20, 80),
        fuelLeft = 10
      )
    }

    val speedTolerance = math.sqrt(2.0 * world.gravity.magnitude * bound.height) / 2
    val taskParams = TaskParams(world, bound, availableActions, initStates,
      hitSpeedTolerance = speedTolerance, rotationTolerance = 0.40,
      rewardFunction = RewardFunction.LinearProduct(
        driftTolerance = bound.width / 3,
        rotationTolerance = 1.0 / 2,
        speedTolerance = speedTolerance,
        totalTime = 10.0))

    val terminateFunc = Simulator.standardTerminateFunc(taskParams.worldBound,
      taskParams.hitSpeedTolerance, taskParams.rotationTolerance) _

    def balanceThrustPolicy(state: State): (Action, PolicyInfo) = {
      val thrust = world.gravity.magnitude / world.maxThrust - 0.1
      val rotSpeed = if (state.rotation.angle > 0) -1.0 else 1.0
      Action(rotationSpeed = rotSpeed, thrust) -> NoInfo
    }

    val train = new DoubleQTraining(taskParams,
      trainParams = params,
      initPolicy = balanceThrustPolicy,
      sampleInitState = sampleInitState
    )

    var trainingReward = 0.0
    val trainingRewardDelay = 1.0 / 20

    import ammonite.ops._

    val resultsDir = pwd / "results" / (TimeTools.numericalDateTime() + s"[ioId=$ioId]")
    val testCurveFile = (resultsDir / "testCurve.txt").toString()
    val trainCurveFile = (resultsDir / "trainCurve.txt").toString()
    val visualizerDataPath = (resultsDir / "visualizerData.serialized").toString()

    val visualizer: TraceRecorder = if(useGUI) new TraceVisualizer(s"ioId=$ioId", bound) else new FakeVisualizer(bound)
    var highestTestScore = 0.0

    def saveAllData(nameTag: String, oldNet: MultiLayerNetwork, newNet: MultiLayerNetwork): Unit ={
      println("Save model...")
      val newNetFile = new File((resultsDir / s"newNet-$nameTag.deep4j").toString())
      val oldNetFile = new File((resultsDir / s"oldNet-$nameTag.deep4j").toString())
      ModelSerializer.writeModel(newNet, newNetFile, true)
      ModelSerializer.writeModel(oldNet, oldNetFile, true)
      FileInteraction.writeToFile((resultsDir/ "highest-score.txt").toString())(highestTestScore.toString)
      visualizer.saveData(visualizerDataPath)
    }

    def checkPointAction(checkPoint: CheckPoint): Unit = {
      import checkPoint._

      val needInit = iteration == 0


      if (iteration % 200 == 0) {
        val policy = NetworkModel.networkToPolicy(newNet, availableActions, exploreRate = None) _
        val scores = initStates.zipWithIndex.map { case (initState, i) =>
          val sim = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
          val r = taskParams.rewardFunction.reward(sim.ending)
          visualizer.addTrace(s"Test[iter=$iteration, taskId=$i]", sim, r)
          r
        }
        val testReward = SimpleMath.mean(scores)
        highestTestScore = math.max(highestTestScore, testReward)
        visualizer.addTestScore(iteration, testReward)
        FileInteraction.writeToFile(testCurveFile, append = true)(s"$iteration, $testReward\n")
        println(s"Test Avg Reward[iter=$iteration]: $testReward")
      }
      newSims.map(s => taskParams.rewardFunction.reward(s.ending)).foreach { r =>
        trainingReward = trainingReward * (1.0 - trainingRewardDelay) + trainingRewardDelay * r
      }
      visualizer.addTrainScore(iteration, trainingReward)
      FileInteraction.writeToFile(trainCurveFile, append = true)(s"$iteration, $trainingReward\n")
      println(s"Exp Weighted Reward[iter=$iteration]: $trainingReward")

      if (needInit) {
        visualizer.initializeFrame()
      }
      if(visualizer.seeCurve){
        visualizer.redisplayData()
      }
      if(iteration % 2000 == 0){
        saveAllData(iteration.toString, oldNet, newNet)
      }
    }

    def shouldContinue(): Boolean = {
      visualizer.shouldContinue
    }

    train.train(20000+1,
      exploreRateFunc = e => params.exploreAmount / (1.0 + e.toDouble / params.exploreDecay), resultsDir, checkPointAction, shouldContinue)
    visualizer.close()
    println(s"task $ioId finished")
  }

  def testUI(args: Array[String]): Unit = {
    val bound = WorldBound(width = 250, height = 150)
    val initState = State(Vec2(50, 100), Vec2.left * 10, Rotation2(0.5), goalX = 10, fuelLeft = 10)


    val rand = new Random(1)
    def randomPolicy(state: State): (Action, PolicyInfo) = {
      Action(rotationSpeed = 2 * rand.nextDouble() - 1, thrust = rand.nextDouble()) -> NoInfo
    }

    val simResult = Simulator(initState,
      world,
      terminateFunc = Simulator.standardTerminateFunc(bound, hitSpeedTolerance = 5, rotationTolerance = 0.15 * math.Pi)
    ).simulateUntilResult(randomPolicy)

    println(s"Ending: ${simResult.ending}")


    Rx.unsafe {

      val scPanel = new StateWithControlPanel(bound, IS("testUI" -> simResult), IS(0.0), 0, 0)


      scPanel.jPanel.setPreferredSize(new Dimension(800, 500))

      val frame = new JFrame()
      frame.setContentPane(scPanel.jPanel)
      frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
      frame.pack()
      frame.setVisible(true)
    }
  }
}
