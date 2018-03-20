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
import ammonite.ops.Path
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

  def getModelSizes(layers: Int, baseNeurons: Int): IS[Int] = {
    var neurons = baseNeurons
    observationLen +: (0 until layers).map{_ =>
      val b = neurons
      neurons /= 2
      b
    }
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

    val c1 = config.layer(sizes.length-1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .nIn(sizes.last).nOut(1)
      .activation(Activation.RELU).build())
      .pretrain(false).backprop(true)
      .build()

    val net = new MultiLayerNetwork(c1)
    net.init()
    net
  }

  def stateToDoubles(state: State): Array[Double] = {
    import state._
    Array(pos.x/100, pos.y/100, velocity.x/20, velocity.y/20,
      rotation.angle, goalX/100, fuelLeft)
  }

  def stateActionArray(state: State, action: Action): Array[Double] = {
    stateToDoubles(state) ++ actionToDoubles(action)
  }

  @inline
  def actionToDoubles(action: Action): Array[Double] = {
    Array(action.rotationSpeed, action.thrust)
  }

  def networkToPolicy(net: MultiLayerNetwork, availableActions: IS[Action], exploreRate: Option[(Double, Random)])(state: State): (Action, PolicyInfo) = {
    exploreRate.foreach{ case (eRate, rand) =>
      if(rand.nextDouble() < eRate)
        return (SimpleMath.randomSelect(rand)(availableActions), ExplorationInfo)
    }

    val inputs = Nd4j.create(availableActions.map(a => stateActionArray(state, a)).toArray)
    val (qValue, idx) = net.output(inputs, false).data().asDouble().zipWithIndex.maxBy(_._1)

    availableActions(idx) -> NetPolicyInfo(qValue)
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

@deprecated
object StateSampling{
  case class SamplingParams(initXRange: (Double, Double),
                            initYRange: (Double, Double), goalXRange: (Double, Double), fuelRange: (Double, Double))
}

object DoubleQTraining{


  case class Observation(tensor: Array[Double], newState: Option[Array[Double]], reward: Double){
    override def toString: String = tensor.mkString("["," ,","]") + s" -> $reward"
  }

  case class CheckPoint(iteration: Int, oldNet: MultiLayerNetwork, newNet: MultiLayerNetwork,
                        newSims: mutable.ListBuffer[FullSimulation])
}

import spaceY.DoubleQTraining._

class DoubleQTraining(taskParams: TaskParams,
                      trainParams: TrainingParams,
                      initPolicy: RPolicy,
                      sampleInitState: Random => State) {
  import taskParams._
  import trainParams._

  val terminateFunc: State => Option[SimulationEnding] =
    Simulator.standardTerminateFunc(worldBound, hitSpeedTolerance, rotationTolerance)

  val rand = new Random(seed+1)

  def sampleObservations(policy: RPolicy, sampleNum: Int, println: Any => Unit): (ListBuffer[FullSimulation], ListBuffer[Observation]) = {

    val observations = mutable.ListBuffer[Observation]()
    val simulations = mutable.ListBuffer[FullSimulation]()
    var runs, landed = 0
    var nOb = 0
    while(nOb < sampleNum) {
      System.out.println(s"sampling progress: $nOb / $sampleNum")
      val initState = sampleInitState(rand)

      val sim@FullSimulation(trace, ending) = new Simulator(initState, world, terminateFunc).simulateUntilResult(policy)
      simulations+=sim
      if(ending.isInstanceOf[Landed]){
        landed += 1
      }

      val reverseTrace = trace.reverse
      val last = stateToDoubles(reverseTrace.head._1) ++ actionToDoubles(reverseTrace.head._2._1)
      observations += Observation(last, None, rewardFunction.reward(ending))

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

  def train(maxIter: Int, exploreRateFunc: Int => Double, resultsDir: Path,
            checkPointAction: CheckPoint => Unit) = {

    FileInteraction.runWithAFileLogger(resultsDir / "log.txt") { logger =>
      import logger._

      println("Parameters:")
      println(trainParams.show)

      val start = System.nanoTime()

      val newNet = createModel(seed, modelParams)
      val oldNet = newNet.clone()

      println("Model summery:")
      println(newNet.summary())

      println("sampling init observations")
      val replayBuffer = mutable.Queue[Observation](
        sampleObservations(initPolicy, replayBufferSize ,println)._2: _*)
      println("sampling finished")


      for (iter <- 0 until maxIter) {
        val timePassed = java.time.Duration.ofNanos(System.nanoTime() - start).getSeconds
        System.out.println(s"Iteration $iter starts [$timePassed s]")

        val exploreRate = exploreRateFunc(iter)

        val (newSims, newObs) = sampleObservations(networkToPolicy(newNet, availableActions, Some(exploreRate, rand)),
          updateDataNum, println)
        (0 until updateDataNum).foreach {
          _ => replayBuffer.dequeue()
        }
        replayBuffer.enqueue(newObs: _*)

        for (b <- 0 until batchesPerDataCollect) {
          val (features, labels) = getBatch(replayBuffer, oldNet, newNet, batchSize)
          newNet.fit(new DataSet(features, labels))

          System.out.print(".")
        }
        System.out.println("")
//        if (iter % copyInterval == 0) {
//          oldNet.setParams(newNet.params())
//        }
        val netReplaceRate: Double = 1.00 / copyInterval
        val newNetParams = oldNet.params().mul(1.0 - netReplaceRate).add(newNet.params().mul(netReplaceRate))
        oldNet.setParams(newNetParams)

        checkPointAction(CheckPoint(iter, oldNet, newNet, newSims))
      }
    }
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
        case Some(newS) => ob.reward + gamma * valueEstimation(oldNet, newNet, newS) //fixme: parallelism
      })
    }.toArray
    (Nd4j.create(inputs), Nd4j.create(outputs))
  }

}
