package rl

import rl.RLProblem.Reward
import rl.WindyGridWorld._
import spaceY.IS

import scala.util.Random

object WindyGridWorld{
  case class VecI2(x: Int, y: Int){
    def + (v: VecI2): VecI2 = {
      VecI2(x + v.x, y+v.y)
    }
  }

  def main(args: Array[String]): Unit = {
    import collection.mutable

    val problem = WindyGridWorld(
      width = 10, height = 7,
      windStrength = IS(0,0,0,1,1,1,2,2,1,0),
      initPos = VecI2(0, 3),
      goal = VecI2(7,3))

    val rand = new Random(1)
    val sarsa = new Sarsa(problem, exploreRate = t => 0.1 / (1 + t / 1000.0), learningRate = t => 0.5 , rand)
    val qMap: sarsa.QMap = mutable.HashMap()
    for(s <- problem.allNonTermStates; a <- problem.availableActions(s)){
      qMap(s -> a) = 0.0
    }
    val traces = sarsa.runExpectedSarsa(qMap, startTime = 0).take(2000).toIndexedSeq

    var timeStep, episode = 0
    var points: IS[(Double, Double)] = IS()
    traces.foreach{ trace =>
      println(trace.length)
      timeStep += trace.length
      episode += 1
      points :+= (timeStep.toDouble -> episode.toDouble)
    }


    val plt = GamblerProblem.doublePlot(points, "Results")
    import breeze.plot.Figure
    val f = Figure()
    f.subplot(0) += plt
    f.refresh()
  }
}

case class WindyGridWorld(width: Int, height: Int,
                          windStrength: IS[Int], initPos: VecI2, goal: VecI2) extends DynamicRLProblem[VecI2, VecI2]{
  def restrict(v: VecI2): VecI2 = {
    val x = if(v.x<0) 0 else if(v.x>=width) width-1 else v.x
    val y = if(v.y<0) 0 else if(v.y>=height) height-1 else v.y
    VecI2(x,y)
  }

  def sampleTransition(s: VecI2, a: VecI2, random: Random): (VecI2, Reward) = {
    val s1 = restrict(s + a + VecI2(0,windStrength(s.x)))
    (s1, -1.0)
  }

  val allNonTermStates: IS[VecI2] = {
    for(x <- 0 until width; y <- 0 until height;
        s = VecI2(x,y) if !isTermState(s)
    ) yield s
  }

  def isTermState(s: VecI2): Boolean = s == goal

  def discountRate: Reward = 1.0

  def availableActions(s: VecI2): IS[VecI2] = IS(VecI2(-1,0), VecI2(1,0), VecI2(0,1), VecI2(0,-1))

  def sampleInitState(random: Random): VecI2 = initPos
}

