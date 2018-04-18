package rl
import rl.RLProblem.{Prob, Reward}
import spaceY.IS

import scala.collection.mutable


case class GamblerProblem(pHead: Double, winGoal: Int) extends StaticRLProblem[Int, Int] {
  val allNonTermStates: IS[Int] ={
    (1 until winGoal).reverse
  }

  def isTermState(s: Int): Boolean = s == 0 || s == winGoal

  def discountRate: Prob = 1.0

  def availableActions(s: Int): IS[Int] = {
    1 to math.min(s, winGoal - s)
  }

  def transition(s: Int, a: Int): IS[(Prob, Int, Reward)] = {
    IS(
      (pHead, s + a, if(s+a == winGoal) 1.0 else 0.0),
      (1.0-pHead, s - a, 0.0)
    )
  }
}

object GamblerProblem{
  def doublePlot(xys: IS[(Double, Double)], name: String) = {
    import breeze.linalg._
    import breeze.plot._

    val (xs,ys) = xys.sorted.unzip
    val xVec = DenseVector(xs:_*)
    val yVec = DenseVector(ys:_*)
    plot(xVec, yVec, name = name)
  }

  def displayPolicy(policy: Map[Int, Int], vMap: Map[Int, Double]): Unit ={
    import breeze.plot._

    val figure = Figure()
    figure.subplot(1, 2, 0) += doublePlot(vMap.toIndexedSeq.map(p => p._1.toDouble -> p._2), "Values")
    figure.subplot(1, 2, 1) += doublePlot(policy.toIndexedSeq.map(p => p._1.toDouble -> p._2.toDouble), "Policy")
    figure.refresh()
  }

  def main(args: Array[String]): Unit = {
    val winGoal = 100
    val prob = GamblerProblem(pHead = 0.25, winGoal = winGoal)
    def initPolicy(s: Int): Int = 1

    val vMap = mutable.HashMap((1 until winGoal).map(x => x -> 0.0) :_*)

//    val policy = DynamicProgramming(prob, tolerance = 0.001).policyImprove(
//      initPolicy,
//      vMap,
//      (_, _) => Unit
//    )

    val dp = DynamicProgramming(prob, tolerance = 1e-10)

    import breeze.plot.Series
    var plots = IS[(Series, Series)]()
    def displayVMap(vMap: dp.ValueMap): Unit = {
      val policy = dp.extractPolicy(vMap)
      val plt = (
        doublePlot(vMap.toIndexedSeq.map(p => p._1.toDouble -> p._2), "Values"),
        doublePlot(policy.toIndexedSeq.map(p => p._1.toDouble -> p._2.toDouble), "Policy")
      )
      plots :+= plt
    }

    dp.valueIteration(vMap, displayVMap)

    var select = 0

    import breeze.plot.Figure
    val figure = Figure()
    figure.refresh()
    def update(): Unit = {
      val (p1, p2) = plots(select)
      figure.clear()
      figure.subplot(1, 2, 0) += p1
      figure.subplot(1, 2, 1) += p2
      figure.refresh()
    }

    import javax.swing._
    import spaceY.GUI._
    val slider = new JSlider()
    slider.setMinimum(0)
    slider.setMaximum(plots.length-1)
    slider.addChangeListener(_ => {
      select = slider.getValue
      update()
    })

    val controlPane = panel(horizontal = true){slider}
    new JFrame("Control"){
      setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
      setContentPane(controlPane)
      pack()
      setVisible(true)
    }
  }
}