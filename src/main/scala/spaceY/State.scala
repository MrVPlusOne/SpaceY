package spaceY

import spaceY.Geometry2D.Vec2


case class State(pos: Vec2, velocity: Vec2, rotation: Double, goalX: Double, fuelLeft: Double) {
}

case class Action(rotationSpeed: Double, thrust: Double)

case class World(gravity: Vec2, maxThrust: Double, deltaT: Double){

  def update(state: State, action: Action): State = {
    val newPos = state.pos + state.velocity * deltaT
    val newVelocity = state.velocity + gravity * deltaT +
      (Vec2.up * maxThrust * action.thrust).rotate(state.rotation) * deltaT
    val newRotation = state.rotation + action.rotationSpeed * deltaT
    val newState = State(newPos, newVelocity, newRotation, state.goalX, state.fuelLeft - deltaT * action.thrust)

    newState
  }
}

case class Simulator(initState: State, world: World, terminateFunc: State => Option[SimulationEnding]){
  def simulate(policy: State => Action): Iterator[(State,Action)] = {
    Iterator.iterate(initState -> policy(initState)){ case (s,_) =>
      val a1 = policy(s)
      world.update(s, a1) -> a1
    }
  }

  def simulateUntilResult(policy: State => Action): (Seq[(State,Action)], SimulationEnding) = {
    var ending: SimulationEnding = null
    val steps = simulate(policy).takeWhile{ case (s, a) =>
      terminateFunc(s) match {
        case None => true
        case Some(end) =>
          ending = end
          false
      }
    }.toList
    (steps, ending)
  }
}

object Simulator{
  case class WorldBound(width: Double, height: Double)

  def standardTerminateFunc(bound: WorldBound, hitSpeedTolerance: Double, rotationTolerance: Double)(state: State): Option[SimulationEnding] = {
    if(state.fuelLeft < 0) return Some(Crashed("out of fuel"))
    if(math.abs(state.pos.x) > bound.width/2 || state.pos.y > bound.height) return Some(Crashed("out of bound"))

    if(state.pos.y <= 0){
      if(math.abs(state.rotation) < rotationTolerance && state.velocity.magnitude < hitSpeedTolerance){
        val score = (1.0 - 0.5 * state.rotation / rotationTolerance) * (1.0 - 0.5 * state.velocity.magnitude / hitSpeedTolerance)
        return Some(Landed(state, score))
      }else{
        return Some(Crashed("landing failed"))
      }
    }
    None
  }
}



sealed trait SimulationEnding
case class Landed(state: State, landingScore: Double) extends SimulationEnding
case class Crashed(info: String) extends SimulationEnding
