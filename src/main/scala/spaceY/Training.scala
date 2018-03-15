package spaceY

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import spaceY.Geometry2D.Vec2
import spaceY.Simulator.WorldBound

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Random

class Training(world: World, worldBound: WorldBound, initFuel: Double,
                    hitSpeedTolerance: Double, rotationTolerance: Double) {

  val terminateFunc: State => Option[SimulationEnding] =
    Simulator.standardTerminateFunc(worldBound, hitSpeedTolerance, rotationTolerance)

  val stateLen = 7
  val actionLen = 2
  val observationLen: Int = stateLen + actionLen
  val batchSize = 64
  val seed = 1
  val rand = new Random(seed+1)
  val gamma = 0.999
  val replayBufferSize = 10000

  def stateToDoubles(state: State): Array[Double] = {
    import state._
    Array(pos.x, pos.y, velocity.x, velocity.y,
      SimpleMath.wrapInRange(rotation, 2*math.Pi), goalX, fuelLeft)
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

  def sampleObservations(): ListBuffer[Observation] = {
    def initPolicy(state: State): Action = {
      val thrust = world.gravity.magnitude / world.maxThrust - 0.1 * rand.nextDouble()
      Action(rotationSpeed = 0.0 , thrust)
    }

    val observations = mutable.ListBuffer[Observation]()
    var nOb = 0
    while(nOb < replayBufferSize) {
      println(s"sampling progress: $nOb")
      val initPos = Vec2(
        (worldBound.width / 3) * (2 * rand.nextDouble() - 1),
        worldBound.height / 3 * rand.nextDouble())
      val initV = Vec2(10* rand.nextDouble(), 5*rand.nextDouble())
      val goalX = (worldBound.width / 4) * (2* rand.nextDouble() - 1)
      val initState = State(initPos, initV, rotation = 0, goalX, fuelLeft = initFuel * rand.nextDouble())

      val (trace, ending) = new Simulator(initState, world, terminateFunc).simulateUntilResult(initPolicy)
      val toCollect = replayBufferSize - nOb
      val collected = trace.reverse.take(toCollect)
      val last = stateToDoubles(collected.head._1) ++ actionToDoubles(collected.head._2)
      observations += Observation(last , Reward.endingReward(ending))
      observations ++= collected.tail.map{
        case (s, a) => Observation(stateToDoubles(s) ++ actionToDoubles(a), reward = 0)
      }
      nOb += collected.length
    }

    observations
  }

  case class Observation(tensor: Array[Double], reward: Double){
    override def toString: String = tensor.mkString("["," ,","]") + s" -> $reward"
  }

  case class TrainingState()

  def trainPolicy() = {
    println("sampling init observations")
    val initObs = mutable.Queue[Observation](sampleObservations(): _*)
    println("sampling finished")
    initObs.foreach(println)
  }

  def valueEstimation(net: MultiLayerNetwork, tensor: Array[Double]): Double = {
    ???
  }

  def getNBatch(obs: mutable.Queue[Observation], oldNet: MultiLayerNetwork, n: Int): (INDArray, INDArray) = {
    val toUse = rand.shuffle(obs.toIterator).take(n * batchSize).toArray
    val inputs = toUse.map(_.tensor)
    val outputs = toUse.map{ ob =>
      ob.reward + gamma * valueEstimation(oldNet, ob.tensor)
    }
    (Nd4j.create(inputs), Nd4j.create(outputs))
  }

  def createModel(): MultiLayerNetwork = {
    val numIter = 1
    val learningRate = 0.006
    val sizes = IS(observationLen,128,64,32,1)

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
      .layer(1, newLayer(0))
      .layer(2, newLayer(0))
      .layer(3, newLayer(0))
      .pretrain(false).backprop(true)
      .build()

    val net = new MultiLayerNetwork(config)
    net.init()
    net
  }
}
