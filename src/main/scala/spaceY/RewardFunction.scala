package spaceY

object RewardFunction {

  case class LinearProduct(driftTolerance: Double,
                          rotationTolerance: Double,
                          speedTolerance: Double) extends RewardFunction {

    def reward(ending: SimulationEnding): Double = ending match {
      case Landed(state) =>
        import state._
        import SimpleMath.relu
        import Math.abs
        relu(1.0 - abs(pos.x - goalX)/driftTolerance) * relu(1.0 - rotation.abs/rotationTolerance) * relu(1.0 - velocity.magnitude/speedTolerance)
      case _: Crashed => 0
    }
  }
}

trait RewardFunction{
  def reward(ending: SimulationEnding): Double
}
