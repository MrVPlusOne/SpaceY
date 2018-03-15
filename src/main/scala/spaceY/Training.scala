package spaceY

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.types.FLOAT32

import scala.collection.mutable

case class Training() {

  val stateLen = 7
  val actionLen = 2
  val observationLen: Int = stateLen + actionLen

//  def stateActionToTensor(state: State, action: Action): Tensor = state match {
//    case State(pos, velocity, rotation, goalX, timeLeft) =>
//      val floats = Seq(pos.x, pos.y, velocity.x, velocity.y,
//        SimpleMath.wrapInRange(rotation, 2*math.Pi),
//        goalX, timeLeft,
//        10*action.rotationSpeed,
//        10*action.thrust).map(_.toFloat)
//
//      Tensor(FLOAT32, floats:_*)
//  }

  case class Observation(state: State, action: Action, reward: Double, tensor: Tensor)

  case class TrainingState(output: Output)

  def trainPolicy(initObs: mutable.Queue[Observation]): Iterable[TrainingState] = {
    ???
  }

  def createModel() = {
    import org.platanios.tensorflow.api.learn.layers._
    val input = Input(FLOAT32, Shape(-1, observationLen))

    val layer = Linear("linear_1", 128) >> ReLU("relu_1", 0.1f) >>
        Linear("linear_2", 64) >> ReLU("relu_2", 0.1f) >>
        Linear("linear_3", 32) >> ReLU("relu_3")

    val loss = Mean("loss_mean")

    val seed = 1
    val numIter = 1
    val learningRate = 0.006
    val sizes = IS(128,64,32)

    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(numIter)
      .learningRate(learningRate)
      .updater(Updater.ADAM)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(observationLen)
        .nOut(sizes(0))
        .activation(Activation.RELU))

  }
}
