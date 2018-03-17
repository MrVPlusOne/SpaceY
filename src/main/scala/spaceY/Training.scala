package spaceY

import java.awt.Dimension

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import spaceY.Simulator.{FullSimulation, NoInfo, PolicyInfo, RPolicy}
import org.nd4j.linalg.factory.Nd4j
import spaceY.Geometry2D.Vec2
import spaceY.Simulator.WorldBound

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Random
import NetworkModel._
import javax.swing.{JButton, JFrame}
import rx.{Rx, Var}

object NetworkModel{
  val stateLen = 7
  val actionLen = 2
  val observationLen: Int = stateLen + actionLen

  case class NetPolicyInfo(qValue: Double) extends PolicyInfo {
    def displayInfo = s"Q Value: $qValue"
  }

  case object ExplorationInfo extends PolicyInfo{
    def displayInfo = s"Random Exploration"
  }

  case class ModelParams(sizes: IS[Int] = IS(observationLen, 64, 32, 16),
                         updater: Updater = Updater.NESTEROVS,
                         learningRate: Double = 0.002){
    def show: String = {
      s"""
         |sizes: ${sizes.mkString("[",", ","]")}
         |updater: ${updater.name()}
         |learningRate: $learningRate
       """.stripMargin
    }
  }

  def createModel(seed: Int, params: ModelParams): MultiLayerNetwork = {
    val numIter = 1
    import params._

    def newLayer(id: Int) = {
      new DenseLayer.Builder()
        .nIn(sizes(id))
        .nOut(sizes(id + 1))
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER).build()
    }

    var config = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(numIter)
      .learningRate(learningRate)
      .updater(updater)
      .regularization(true).l2(1e-4)
      .list()

    sizes.indices.init.foreach(i => config = config.layer(i, newLayer(i)))

    val c1 = config.layer(sizes.length, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .nIn(sizes.last).nOut(1)
      .activation(Activation.RELU).build())
      .pretrain(false).backprop(true)
      .build()

    val net = new MultiLayerNetwork(c1)
    net.init()
    net
  }

}

case class TrainingParams(modelParams: ModelParams = ModelParams(),
                          batchSize: Int = 64,
                          batchesPerDataCollect: Int = 20,
                          seed: Int = 1,
                          gamma: Double = 0.999,
                          replayBufferSize: Int = 50*100*10,
                          updateDataNum: Int = 50,
                          copyInterval: Int = 100,
                          threadNum: Int = 1){
  def show: String = {
    s"""
       |modelParams: ${modelParams.show}
       |batchSize: $batchSize
       |batchesPerDataCollect: $batchesPerDataCollect
       |seed: $seed
       |gamma: $gamma
       |replayBufferSize: $replayBufferSize
       |updateDataNum: $updateDataNum
       |copyInterval: $copyInterval
       |threadNum: $threadNum
     """.stripMargin
  }
}

class Training(world: World, worldBound: WorldBound, initFuel: Double,
               initState: State, hitSpeedTolerance: Double, rotationTolerance: Double, params: TrainingParams) {

  val terminateFunc: State => Option[SimulationEnding] =
    Simulator.standardTerminateFunc(worldBound, hitSpeedTolerance, rotationTolerance)
  import params._

  val rand = new Random(seed+1)

  def stateToDoubles(state: State): Array[Double] = {
    import state._
    Array(pos.x/100, pos.y/100, velocity.x/10, velocity.y/10,
      SimpleMath.wrapInRange(rotation+math.Pi, 2 * math.Pi) - math.Pi, goalX/100, fuelLeft)
  }

  def stateActionArray(state: State, action: Action): Array[Double] = {
    stateToDoubles(state) ++ actionToDoubles(action)
  }

  @inline
  def actionToDoubles(action: Action): Array[Double] = {
    Array(action.rotationSpeed, 5*(action.thrust-0.5))
  }

  val availableActions: IS[Action] = {
    for(
      rotate <- IS(1.0, 0.0, -1.0);
      thrust <- 0.0 to 1.0 by 1.0/6
    ) yield Action(rotate, thrust)
  }


  case class SamplingParams(initXRange: (Double, Double),
                            initYRange: (Double, Double), goalXRange: (Double, Double), fuelRange: (Double, Double))

  def sampleInRange(range: (Double, Double)): Double = {
    SimpleMath.linearInterpolate(range._1, range._2)(rand.nextDouble())
  }

  def sampleObservations(policy: RPolicy, sampleNum: Int, params: SamplingParams, println: Any => Unit): (ListBuffer[FullSimulation], ListBuffer[Observation]) = {
    import params._

    val observations = mutable.ListBuffer[Observation]()
    val simulations = mutable.ListBuffer[FullSimulation]()
    var runs, landed = 0
    var nOb = 0
    while(nOb < sampleNum) {
      System.out.println(s"sampling progress: $nOb / $sampleNum")
      val initPos = Vec2(
        sampleInRange(initXRange),
        sampleInRange(initYRange)
      )
      val initV = Vec2(10* rand.nextDouble(), 5*rand.nextDouble())
      val goalX = sampleInRange(goalXRange)
      val initState = State(initPos, initV, rotation = rand.nextDouble() - 0.5, goalX, fuelLeft = sampleInRange(fuelRange))

      val sim@FullSimulation(trace, ending) = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
      simulations+=sim
      if(ending.isInstanceOf[Landed]){
        landed += 1
      }

      val reverseTrace = trace.reverse
      val last = stateToDoubles(reverseTrace.head._1) ++ actionToDoubles(reverseTrace.head._2._1)
      observations += Observation(last, None, Reward.endingReward(ending))

      val newTrace = mutable.ListBuffer[Observation]()
      for(i <- 1 until reverseTrace.length){
        val (s1, _) = reverseTrace(i-1)
        val (s0, (a0, _)) = reverseTrace(i)
        newTrace += Observation(stateActionArray(s0, a0), Some(stateToDoubles(s1)), 0)
      }

      val collected = rand.shuffle(newTrace).take(sampleNum - nOb - 1)
      observations ++= collected
      nOb += collected.length + 1
      runs += 1
    }

    println(s"Sample landing rate: ${landed.toDouble / runs}")
    (simulations, observations)
  }

  case class Observation(tensor: Array[Double], newState: Option[Array[Double]], reward: Double){
    override def toString: String = tensor.mkString("["," ,","]") + s" -> $reward"
  }

  case class TrainingState()

  def train(maxIter: Int, exploreRateFunc: Int => Double) = {
    import rx.Ctx.Owner.Unsafe._

    val testTraces: Var[IS[FullSimulation]] = {
      val evaluations = Var(IS[FullSimulation]())

      val placeholder = GUI.panel(horizontal = false)()
      val dataButton = new JButton("Fetch data")
      val frame = new JFrame()
      frame.setContentPane(
        GUI.panel(horizontal = false)(
          placeholder,
          GUI.panel(horizontal = true)(dataButton, GUI.panel(horizontal = false)())
        )
      )
      var oldSCPanel: Option[StateWithControlPanel] = None

      def replaceUI(): Unit = {
        val newP = oldSCPanel match {
          case None =>
            new StateWithControlPanel(worldBound, evaluations.now, 0, 0){
              jPanel.setPreferredSize(new Dimension(600, ((worldBound.height/worldBound.width)*600).toInt))
            }
          case Some(op) =>
            val np = new StateWithControlPanel(worldBound, evaluations.now, op.simulation, op.step)
            np.jPanel.setPreferredSize(op.jPanel.getSize)
            op.stopTracking()
            np
        }

        placeholder.removeAll()
        placeholder.add(newP.jPanel)
        frame.pack()

        oldSCPanel = Some(newP)
      }

      evaluations.triggerLater {
        if (oldSCPanel.isEmpty) {
          frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
          frame.setVisible(true)
          replaceUI()
        }
      }

      dataButton.addActionListener(_ => {
        replaceUI()
      })

      evaluations
    }

    val logFile = s"results/${TimeTools.numericalDateTime()}"
    FileInteraction.runWithAFileLogger(logFile) { logger =>
      import logger._

      println("Parameters: ")
      println(params.show)

      val start = System.nanoTime()

      def initPolicy(state: State): (Action, PolicyInfo) = {
        val thrust = world.gravity.magnitude / world.maxThrust - 0.1 * rand.nextDouble()
        Action(rotationSpeed = 0.0, thrust) -> NoInfo
      }

      println("sampling init observations")
      val initParams = SamplingParams(
        initXRange = (-worldBound.width / 3, worldBound.width / 3),
        initYRange = (0.0, worldBound.height / 3),
        goalXRange = (-worldBound.width / 4, worldBound.width / 4),
        fuelRange = (5, 15)
      )
      val replayBuffer = mutable.Queue[Observation](
        sampleObservations(initPolicy, replayBufferSize, initParams ,println)._2: _*)
      println("sampling finished")

      val newNet = createModel(seed, modelParams)
      val oldNet = newNet.clone()


      for (epoch <- 0 until maxIter) {
        val timePassed = java.time.Duration.ofNanos(System.nanoTime() - start).getSeconds
        println(s"Epoch $epoch starts [$timePassed s]")

        //      val exploreRate = 1.0/(4.0+epoch.toDouble/10)
        val exploreRate = exploreRateFunc(epoch)

        val trainParams = SamplingParams(
          initXRange = (-worldBound.width / 3, worldBound.width / 3),
          initYRange = (0.1 * worldBound.height, 0.8 * worldBound.height),
          goalXRange = (-worldBound.width / 4, worldBound.width / 4),
          fuelRange = (0.1 * initFuel, initFuel)
        )

        val (newSims, newObs) = sampleObservations(networkToPolicy(newNet, Some(exploreRate)),
          updateDataNum, trainParams, println)
        (0 until updateDataNum).foreach {
          _ => replayBuffer.dequeue()
        }
        replayBuffer.enqueue(newObs: _*)

        for (b <- 0 until batchesPerDataCollect) {
          val (features, labels) = getBatch(replayBuffer, oldNet, newNet, batchSize)
          newNet.fit(new DataSet(features, labels))

          print(".")
        }
        println("")
        if (epoch % copyInterval == 0) {
          oldNet.setParams(newNet.params())
        }
        val netReplaceRate: Double = 1.00 / copyInterval
        val newNetParams = oldNet.params().mul(1.0 - netReplaceRate).add(newNet.params().mul(netReplaceRate))
        oldNet.setParams(newNetParams)

        if (epoch % 10 == 0) {
          if (epoch % 100 == 0) {
            val policy = networkToPolicy(newNet, exploreRate = None) _
            val sim@FullSimulation(_, ending) = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
            println(s"Test Ending: $ending")
            println(s"Test Reward: ${Reward.endingReward(ending)}")
            testTraces() = testTraces.now :+ sim
          } else {
            val sim@(FullSimulation(_, ending)) = newSims.maxBy(s => Reward.endingReward(s.ending))

            println(s"Best Ending: $ending")
            println(s"Best Reward: ${Reward.endingReward(ending)}")
            testTraces() = testTraces.now :+ sim
          }
        }
      }

    }
  }

  def networkToPolicy(net: MultiLayerNetwork, exploreRate: Option[Double])(state: State): (Action, PolicyInfo) = {
    exploreRate.foreach{ eRate =>
      if(rand.nextDouble() < eRate)
        return (SimpleMath.randomSelect(rand)(availableActions), ExplorationInfo)
    }

    val inputs = Nd4j.create(availableActions.map(a => stateActionArray(state, a)).toArray)
    val (qValue, idx) = net.output(inputs, false).data().asDouble().zipWithIndex.maxBy(_._1)

    availableActions(idx) -> NetPolicyInfo(qValue)
  }


  def valueEstimation(oldNet: MultiLayerNetwork, newNet: MultiLayerNetwork, state: Array[Double]): Double = {

    val inputs = Nd4j.create(availableActions.toArray.map { a =>
      state ++ actionToDoubles(a)
    })

    val actionId = SimpleMath.maxId(newNet.output(inputs, false).data().asDouble())
    oldNet.output(inputs.getRow(actionId), false).getDouble(0)
  }

  def getBatch(obs: mutable.Queue[Observation], oldNet: MultiLayerNetwork, newNet: MultiLayerNetwork, batchSize: Int): (INDArray, INDArray) = {
    val toUse = Array.fill(batchSize){
      val i = rand.nextInt(replayBufferSize)
      obs(i)
    }
    val inputs = toUse.map(_.tensor)
    val outputs = SimpleMath.parallelMap(toUse, threadNum){ ob =>
      Array(ob.newState match {
        case None => ob.reward
        case Some(newS) => ob.reward + gamma * valueEstimation(oldNet, newNet, newS) //fixme
      })
    }.toArray
    (Nd4j.create(inputs), Nd4j.create(outputs))
  }

}
