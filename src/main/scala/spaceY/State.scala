package spaceY

import spaceY.Geometry2D.{Vec2, Rotation2}
import spaceY.Simulator.{FullSimulation, PolicyInfo, RPolicy, WorldBound}

/**
  * @param fuelLeft Time left! Will change this name in the future
  */
case class State(pos: Vec2, velocity: Vec2, rotation: Rotation2, goalX: Double, fuelLeft: Double) {

  override def toString: String = {
    s"State(F: %.1f, P: $pos, V: $velocity, R: %s, G: %.1f)".format(fuelLeft, rotation, goalX)
  }
}

case class Action(rotationSpeed: Double, thrust: Double){
  override def toString: String = {
    "Action(RSpeed: % .2f, Thrust: % .2f)".format(rotationSpeed, thrust)
  }
}

case class World(gravity: Vec2, maxThrust: Double, deltaT: Double){

  def update(state: State, action: Action): State = {
    val newPos = state.pos + state.velocity * deltaT
    val newVelocity = state.velocity + gravity * deltaT +
      (Vec2.up * maxThrust * action.thrust).rotate(state.rotation) * deltaT
    val newRotation = state.rotation.rotate(action.rotationSpeed * deltaT)
    val newState = State(newPos, newVelocity, newRotation, state.goalX, state.fuelLeft - deltaT)

    newState
  }
}

case class Simulator(initState: State, world: World, terminateFunc: State => Option[SimulationEnding]){
  def simulate(policy: RPolicy): Iterator[(State,(Action,PolicyInfo))] = {
    Iterator.iterate(initState -> policy(initState)){ case (s,_) =>
      val (a1, info) = policy(s)
      (world.update(s, a1), (a1, info))
    }
  }

  def simulateUntilResult(policy: RPolicy): FullSimulation = {
    var ending: SimulationEnding = null
    val steps = simulate(policy).takeWhile{ case (s, _) =>
      terminateFunc(s) match {
        case None => true
        case Some(end) =>
          ending = end
          false
      }
    }.toIndexedSeq
    FullSimulation(steps, ending)
  }
}

object Simulator{

  type RPolicy = State => (Action, PolicyInfo)

  trait PolicyInfo{
    def displayInfo: String
  }

  case object NoInfo extends PolicyInfo{
    def displayInfo: String = "No Info"
  }

  case class WorldBound(width: Double, height: Double)

  case class FullSimulation(trace: IS[(State, (Action, PolicyInfo))], ending: SimulationEnding)

  def standardTerminateFunc(bound: WorldBound, hitSpeedTolerance: Double, rotationTolerance: Double)(state: State): Option[SimulationEnding] = {
    if(state.fuelLeft < 0) return Some(Crashed(state, "out of fuel"))
    if(math.abs(state.pos.x) > bound.width/2 || state.pos.y > bound.height) return Some(Crashed(state, "out of bound"))

    if(state.pos.y <= 0){
      if(state.rotation.abs < rotationTolerance && state.velocity.magnitude < hitSpeedTolerance){
        return Some(Landed(state))
      }else{
        return Some(Crashed(state, "landing failed"))
      }
    }
    None
  }
}



sealed trait SimulationEnding
case class Landed(state: State) extends SimulationEnding
case class Crashed(state: State, info: String) extends SimulationEnding


case class TaskParams(
                       world: World,
                       worldBound: WorldBound,
                       availableActions: IS[Action],
                       initStates: IS[State],
                       hitSpeedTolerance: Double,
                       rotationTolerance: Double,
                       rewardFunction: RewardFunction
                     ){
  override def toString: String = {
    s"""
       |world: $world
       |worldBound: $worldBound
       |availableActions: $availableActions
       |initStates: $initStates
       |hitSpeedTolerance: $hitSpeedTolerance
       |rotationTolerance: $rotationTolerance
       |rewardFunction: $rewardFunction
     """.stripMargin
  }
}