package rl
import rl.RLProblem.{Prob, Reward}
import spaceY.IS

import scala.util.Random

object CarRentalProblem {
  val maxCarNum = 20
  val maxMove = 5
  val lambdaRental1 = 3
  val lambdaRental2 = 4
  val lambdaReturn1 = 3
  val lambdaReturn2 = 2

  case class State(n1: Int, n2: Int){
    require(0 <= n1 && n1 <= maxCarNum)
    require(0 <= n2 && n2 <= maxCarNum)
  }

  case class Action(move: Int){
    require(-maxMove <= move && move <= maxMove)
  }

  def discountRate: Double = 0.9

  def availableActions(s: State): IS[Action] = {
    (-s.n2 to s.n1).map(Action.apply)
  }

  def transition(s: State, a: Action, rand: Random): IS[(Prob, State, Reward)] = {
    val cost = math.abs(a.move) * 2
    val s1 = State(s.n1 - a.move, math.min(s.n2 + a.move, maxCarNum))
    ???
  }
}
