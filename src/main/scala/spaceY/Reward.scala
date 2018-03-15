package spaceY

import spaceY.Geometry2D.Vec2

object Reward {
  def endingReward(ending: SimulationEnding): Double = ending match {
    case Landed(goalX, posX, fuelLeft) => 10 + fuelLeft + math.max(0, 100 - math.abs(goalX - posX))
    case _: Crashed => 0
  }
}
