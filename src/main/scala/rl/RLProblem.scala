package rl

import rl.RLProblem._
import spaceY.{IS, SimpleMath}

import scala.collection.mutable.ListBuffer
import scala.util.Random

object RLProblem{
  type Prob = Double
  type Reward = Double
  type Time = Int
}

trait RLProblem[State, Action] {
  type Policy = State => Action

  def allNonTermStates: IS[State]

  def isTermState(s: State): Boolean

  def discountRate: Double

  def availableActions(s: State): IS[Action]

}

/** RL problem with full knowledge of the transition distribution */
trait StaticRLProblem[State, Action] extends RLProblem[State, Action]{
  def transition(s: State, a: Action): IS[(Prob, State, Reward)]
}

trait DynamicRLProblem[State, Action] extends RLProblem[State, Action] {
  def sampleTransition(s: State, a: Action, random: Random): (State, Reward)

  def sampleInitState(random: Random): State
}


import collection.mutable

case class DynamicProgramming[State, Action](problem: StaticRLProblem[State, Action],
                                             tolerance: Double){
  import problem._

  type ValueMap = mutable.HashMap[State, Double]

  def qValue(s: State, a: Action, vMap: ValueMap) = {
    transition(s, a).map{
      case (p, s1, r) =>
        val v1 = if(isTermState(s1)) 0.0 else vMap(s1)
        (r + discountRate * v1) * p
    }.sum
  }

  def evalPolicy(policy: Policy, vMap: ValueMap): Unit= {
    print(".")
    val delta = allNonTermStates.map{ s =>
      val v = vMap(s)
      vMap(s) = qValue(s, policy(s), vMap)
      math.abs(vMap(s) - v)
    }.max
    if(delta > tolerance) evalPolicy(policy, vMap)
    else{
      println("")
    }
  }

  def policyImprove(oldPolicy: Policy, initVMap: ValueMap, displayPolicy: (Policy, ValueMap) => Unit): Map[State, Action] = {
    val vMap = initVMap
    evalPolicy(oldPolicy, vMap)

    var policyImproved = false
    val newPolicy = allNonTermStates.map{ s =>
      val (newAction, newV) = availableActions(s).map(a => a -> qValue(s, a, vMap)).maxBy(_._2)
      if(newV > vMap(s)){
        policyImproved = true
      }
      s -> newAction
    }.toMap

    displayPolicy(newPolicy, vMap)
    if(policyImproved){
      policyImprove(newPolicy.apply, vMap, displayPolicy)
    }else{
      newPolicy
    }
  }

  def valueIteration(vMap: ValueMap, displayVMap: ValueMap => Unit): Unit = {
    displayVMap(vMap)
    val maxDelta = allNonTermStates.map{ s =>
      val (_, newV) = availableActions(s).map(a => a -> qValue(s, a, vMap)).maxBy(_._2)
      val delta = math.abs(newV - vMap(s))
      vMap(s) = newV
      delta
    }.max

    if(maxDelta > tolerance)
      valueIteration(vMap, displayVMap)
  }

  def extractPolicy(valueMap: ValueMap): Map[State, Action] = {
    allNonTermStates.map{ s => s -> availableActions(s).maxBy{ a => qValue(s, a, valueMap)} }.toMap
  }
}


/** The Sarsa on-policy Temporal-difference control algorithm */
class Sarsa[S,A](problem: DynamicRLProblem[S,A],
                 exploreRate: Int => Double,
                 learningRate: Int => Double, random: Random){
  import problem._

  type QMap = mutable.HashMap[(S,A), Double]

  def policy(qMap: QMap, explore: Double)(s: S): A = {
    if(random.nextDouble() < explore){
      SimpleMath.randomSelect(random)(availableActions(s))
    }else{
      availableActions(s).maxBy{a => qMap(s->a)}
    }
  }


  def simulateAnEpisode(s0: S, a0: A, qMap: QMap, estimateState: (S, QMap, Time) => Double, time: Time): ListBuffer[(A, S, Reward)] = {
    val trace = mutable.ListBuffer[(A, S, Reward)]()
    def loop(s0: S, a0: A, time: Time): Unit ={
      val (s1, r) = sampleTransition(s0, a0, random)
      if(isTermState(s1)){
        val delta = r - qMap(s0, a0)
        qMap(s0 -> a0) += learningRate(time) * delta
        trace.append((a0, s1, r))
      }else{
        val a1 = policy(qMap, exploreRate(time))(s1)
        val delta = r + discountRate * estimateState(s1, qMap, time) - qMap(s0, a0)
        qMap(s0 -> a0) += learningRate(time) * delta
        trace.append((a0, s1, r))
        loop(s1, a1, time + 1)
      }
    }
    loop(s0, a0, time)
    trace
  }

  def runSarsa(qMap: QMap, startTime: Int): Stream[List[(A,S,Reward)]] = {
    def estimateState(s: S, qMap: QMap, time: Time): Double = {
      val a = policy(qMap, exploreRate(time))(s)
      qMap(s -> a)
    }

    val s0 = sampleInitState(random)
    val a0 = policy(qMap, exploreRate(startTime))(s0)
    val trace = simulateAnEpisode(s0, a0, qMap, estimateState, time = startTime)

    Stream.cons(trace.toList, runSarsa(qMap, startTime + trace.length))
  }

  def runExpectedSarsa(qMap: QMap, startTime: Int): Stream[List[(A,S,Reward)]] = {
    def estimateState(s: S,qMap: QMap, time: Time): Double = {
      val explore = exploreRate(time)
      val base = explore / availableActions(s).length
      val greedy = availableActions(s).maxBy{a => qMap(s->a)}
      availableActions(s).map{ a =>
        val weight = if(a == greedy) (1.0 - explore) + base else base
        weight * qMap(s -> a)
      }.sum
    }

    val s0 = sampleInitState(random)
    val a0 = policy(qMap, exploreRate(startTime))(s0)
    val trace = simulateAnEpisode(s0, a0, qMap, estimateState, time = startTime)

    Stream.cons(trace.toList, runSarsa(qMap, startTime + trace.length))
  }

}