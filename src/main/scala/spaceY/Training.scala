package spaceY

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import spaceY.Simulator.FullSimulation
//import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import spaceY.Geometry2D.Vec2
import spaceY.Simulator.WorldBound

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Random

class Training(world: World, worldBound: WorldBound, initFuel: Double,
               initState: State, hitSpeedTolerance: Double, rotationTolerance: Double) {

  val terminateFunc: State => Option[SimulationEnding] =
    Simulator.standardTerminateFunc(worldBound, hitSpeedTolerance, rotationTolerance)

  val stateLen = 7
  val actionLen = 2
  val observationLen: Int = stateLen + actionLen
  val batchSize = 128
  val batchesPerDataCollect = 10
  val seed = 1
  val rand = new Random(seed+1)
  val gamma = 0.999
  val replayBufferSize = 50000
  val updateDataNum =  1000

  def stateToDoubles(state: State): Array[Double] = {
    import state._
    Array(pos.x, pos.y, velocity.x, velocity.y,
      SimpleMath.wrapInRange(rotation, 2*math.Pi), goalX, fuelLeft)
  }

  def stateActionTensor(state: State, action: Action): INDArray = {
    import state._
    Nd4j.create(Array(pos.x, pos.y, velocity.x, velocity.y,
      SimpleMath.wrapInRange(rotation, 2*math.Pi), goalX, fuelLeft, action.rotationSpeed, action.thrust))
  }

  def actionToDoubles(action: Action): Array[Double] = {
    Array(action.rotationSpeed, action.thrust)
  }

  val availableActions: IS[Action] = {
    for(
      rotate <- IS(math.Pi, -math.Pi);
      thrust <- 0.0 to 1.0 by 0.1
    ) yield Action(rotate, thrust)
  }


  case class SamplingParams(initXRange: (Double, Double),
                            initYRange: (Double, Double), goalXRange: (Double, Double), fuelRange: (Double, Double))

  def sampleInRange(range: (Double, Double)): Double = {
    SimpleMath.linearInterpolate(range._1, range._2)(rand.nextDouble())
  }

  def sampleObservations(policy: State => Action, sampleNum: Int, params: SamplingParams): ListBuffer[Observation] = {
    import params._

    val observations = mutable.ListBuffer[Observation]()
    var runs, landed = 0
    var nOb = 0
    while(nOb < sampleNum) {
      println(s"sampling progress: $nOb / $sampleNum")
      val initPos = Vec2(
        sampleInRange(initXRange),
        sampleInRange(initYRange)
      )
      val initV = Vec2(10* rand.nextDouble(), 5*rand.nextDouble())
      val goalX = sampleInRange(goalXRange)
      val initState = State(initPos, initV, rotation = 0, goalX, fuelLeft = sampleInRange(fuelRange))

      val FullSimulation(trace, ending) = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
      if(ending.isInstanceOf[Landed]){
        landed += 1
      }
      val toCollect = sampleNum - nOb
      val collected = trace.reverse.take(toCollect)
      val last = stateToDoubles(collected.head._1) ++ actionToDoubles(collected.head._2)
      observations += Observation(last , Reward.endingReward(ending))
      observations ++= collected.tail.map{
        case (s, a) => Observation(stateToDoubles(s) ++ actionToDoubles(a), reward = 0)
      }
      nOb += collected.length
      runs += 1
    }

    println(s"Sample landing rate: ${landed.toDouble / runs}")
    observations
  }

  case class Observation(tensor: Array[Double], reward: Double){
    override def toString: String = tensor.mkString("["," ,","]") + s" -> $reward"
  }

  case class TrainingState()

  def train(maxIter: Int, netReplaceRate: Double, exploreRateFunc: Int => Double) = {
    def initPolicy(state: State): Action = {
      val thrust = world.gravity.magnitude / world.maxThrust - 0.1 * rand.nextDouble()
      Action(rotationSpeed = 0.0 , thrust)
    }

    println("sampling init observations")
    val initParams = SamplingParams(
      initXRange = (-worldBound.width/3, worldBound.width/3),
      initYRange = (0.0, worldBound.height/3),
      goalXRange = (-worldBound.width/4, worldBound.width/4),
      fuelRange = (5, 15)
    )
    val replayBuffer = mutable.Queue[Observation](sampleObservations(initPolicy, replayBufferSize, initParams): _*)
    println("sampling finished")

    val newNet = createModel()
    val oldNet = newNet.clone()
    for(epoch <- 0 until maxIter){
      println(s"Epoch $epoch starts")

//      val exploreRate = 1.0/(4.0+epoch.toDouble/10)
      val exploreRate = exploreRateFunc(epoch)

      val trainParams = SamplingParams(
        initXRange = (-worldBound.width/3, worldBound.width/3),
        initYRange = (0.1 * worldBound.height, 0.8 * worldBound.height),
        goalXRange = (-worldBound.width/4, worldBound.width/4),
        fuelRange = (initFuel * math.max(0, 0.5 - exploreRate), initFuel)
      )

      val newObs = sampleObservations(networkToPolicy(newNet, Some(exploreRate)), updateDataNum, trainParams)
      (0 until updateDataNum).foreach{
        _ => replayBuffer.dequeue()
      }
      replayBuffer.enqueue(newObs :_*)

      for(b <- 0 until batchesPerDataCollect){
        val (features, labels) = getBatch(replayBuffer, oldNet, newNet, batchSize)
        newNet.fit(new DataSet(features, labels))

        val newNetParams = oldNet.params().mul(1.0 - netReplaceRate).add(newNet.params().mul(netReplaceRate))
        oldNet.setParams(newNetParams)

        print(".")
      }
      println()



      val policy = networkToPolicy(newNet, exploreRate = None) _
      val FullSimulation(trace, ending) = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
      println(s"Test Ending: $ending")
      println(s"Test Reward: ${Reward.endingReward(ending)}")
    }

  }

  def networkToPolicy(net: MultiLayerNetwork, exploreRate: Option[Double])(state: State): Action = {
    exploreRate.foreach{ eRate =>
      if(rand.nextDouble() < eRate)
        return SimpleMath.randomSelect(rand)(availableActions)
    }
    availableActions.maxBy{ a =>
      net.output(stateActionTensor(state, a), false).getDouble(0)
    }
  }


  def valueEstimation(oldNet: MultiLayerNetwork, newNet: MultiLayerNetwork, tensor: Array[Double]): Double = {
    val (bestA, _) = availableActions.map{ a =>
      val t = tensor.clone()
      t(stateLen) = a.rotationSpeed
      t(stateLen+1) = a.thrust
      val input = Nd4j.create(t)
      a -> newNet.output(input, false).getDouble(0)
    }.maxBy(_._2)

    val t = tensor.clone()
    t(stateLen) = bestA.rotationSpeed
    t(stateLen+1) = bestA.thrust
    val input = Nd4j.create(t)
    oldNet.output(input, false).getDouble(0)
  }

  def getBatch(obs: mutable.Queue[Observation], oldNet: MultiLayerNetwork, newNet: MultiLayerNetwork, batchSize: Int): (INDArray, INDArray) = {
    val toUse = Array.fill(batchSize){
      val i = rand.nextInt(replayBufferSize)
      obs(i)
    }
    val inputs = toUse.map(_.tensor)
    val outputs = toUse.map{ ob =>
      Array(ob.reward + gamma * valueEstimation(oldNet, newNet, ob.tensor))
    }
    (Nd4j.create(inputs), Nd4j.create(outputs))
  }

  def createModel(): MultiLayerNetwork = {
    val numIter = 1
    val learningRate = 0.0005
    val sizes = IS(observationLen,128,64,32)

    def newLayer(id: Int) = {
      new DenseLayer.Builder()
        .nIn(sizes(id))
        .nOut(sizes(id+1))
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER).build()
    }

    val config = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(numIter)
      .learningRate(learningRate)
      .updater(Updater.ADAM)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, newLayer(0))
      .layer(1, newLayer(1))
      .layer(2, newLayer(2))
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(sizes.last).nOut(1)
        .activation(Activation.RELU).build())
      .pretrain(false).backprop(true)
      .build()

    val net = new MultiLayerNetwork(config)
    net.init()
    net
  }
}
