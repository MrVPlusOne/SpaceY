package spaceY

import java.awt.Dimension

import javax.swing.JFrame
import rx._
import spaceY.DoubleQTraining.CheckPoint
import spaceY.Geometry2D._
import spaceY.Simulator.{FullSimulation, NoInfo, PolicyInfo, WorldBound}

import scala.util.Random

object Playground {

  val deltaT: Double = 1.0/25.0
  val world = World(gravity = Vec2.down*10, maxThrust = 30, deltaT = deltaT)

  def main(args: Array[String]): Unit = {
//    BasicConfigurator.configure()

    val params = TrainingParams()

    val bound = WorldBound(width = 200, height = 150)
    val availableActions: IS[Action] = {
      for(
        rotate <- IS(1.0, 0.0, -1.0);
        thrust <- 0.0 to 1.0 by 1.0/6
      ) yield Action(rotate, thrust)
    }

    val initStates = IS(
      State(pos = Vec2(20, 50), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 20, fuelLeft = 10),
      State(pos = Vec2(20, 125), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 75, fuelLeft = 10),
      State(pos = Vec2(-75, 100), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 75, fuelLeft = 10),
      State(pos = Vec2(0, 125), velocity = Vec2.zero, rotation = Rotation2(-0.65), goalX = 25, fuelLeft = 10),
      State(pos = Vec2(-25, 75), velocity = Vec2(-10,-10), rotation = Rotation2(0), goalX = 25, fuelLeft = 10)
    )

    def sampleInitState(random: Random): State = {
      def between(from: Double, to: Double) = SimpleMath.linearInterpolate(from, to)(random.nextDouble())
      State(
        pos = Vec2(between(-80, 80), between(50,125)),
        velocity = Vec2(between(-20,20), between(-20,20)),
        rotation = Rotation2(SimpleMath.cubic(between(-1.0, 1.0))*math.Pi),
        goalX = between(-20,80),
        fuelLeft = 10
      )
    }

    val taskParams = TaskParams(world, bound, availableActions, initStates,
      hitSpeedTolerance = 30, rotationTolerance = 0.4 * math.Pi,
      rewardFunction = RewardFunction.LinearProduct(
        driftTolerance = bound.width/3,
        rotationTolerance = 1.0/2,
        speedTolerance = math.sqrt(2.0 * world.gravity.magnitude * bound.height)/2))

    val terminateFunc = Simulator.standardTerminateFunc(taskParams.worldBound,
      taskParams.hitSpeedTolerance, taskParams.rotationTolerance) _

    def balanceThrustPolicy(state: State): (Action, PolicyInfo) = {
      val thrust = world.gravity.magnitude / world.maxThrust - 0.1
      Action(rotationSpeed = 0.0, thrust) -> NoInfo
    }

    val train = new DoubleQTraining(taskParams,
      trainParams = params,
      initPolicy = balanceThrustPolicy,
      sampleInitState = sampleInitState
    )

    val visualizer = new TraceVisualizer(IS(), IS(), taskParams.worldBound)
    def checkPointAction(checkPoint: CheckPoint): Unit = {
      import checkPoint._

      val needInit = visualizer.dataNum == 0

      if (iteration % 10 == 0) {
        if (iteration % 100 == 0) {
          val policy = NetworkModel.networkToPolicy(newNet, availableActions, exploreRate = None) _
          val scores = initStates.zipWithIndex.map { case (initState, i) =>
            val sim = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
            val r = taskParams.rewardFunction.reward(sim.ending)
            visualizer.addTrace(s"Test[iter=$iteration, taskId=$i]", sim, r)
            r
          }
          val testReward = SimpleMath.mean(scores)
          println(s"Test Avg Reward[iter=$iteration]: $testReward")
        }
        val avgReward = SimpleMath.mean(newSims.map(s => taskParams.rewardFunction.reward(s.ending)))

        println(s"Avg Reward[iter=$iteration]: $avgReward")
      }

      if(needInit){
        visualizer.redisplayData()
        visualizer.initializeFrame()
      }
    }

//    NetworkModel.createModel()
    import ammonite.ops._

    val resultsDir = pwd / "results" / TimeTools.numericalDateTime()
    train.train(20000,
      exploreRateFunc = e => 0.05/(5+e.toDouble/100), resultsDir, checkPointAction)
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
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.pack()
      frame.setVisible(true)
    }
  }
}
