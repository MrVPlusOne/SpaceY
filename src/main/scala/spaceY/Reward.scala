package spaceY

import spaceY.Geometry2D.Vec2

object Reward {
  def endingReward(ending: SimulationEnding): Double = ending match {
    case Landed(state, score) =>
      import state._
      score * (10 + fuelLeft + math.max(0, 100 - math.abs(goalX - pos.x))) / 100
    case _: Crashed => 0
  }
}
