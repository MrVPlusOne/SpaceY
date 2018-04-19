package rl

import java.awt.geom.Rectangle2D
import java.awt.{Dimension, Graphics, Graphics2D}

import RocketLanding._
import javax.swing.JPanel
import org.jfree.chart.JFreeChart
import rl.RLProblem.Reward
import spaceY._

import scala.collection.mutable
import scala.util.Random

case class RocketFeature(bound: spaceY.Simulator.WorldBound,
                         xDivide: Int, yDivide: Int,
                         vxDivide: IS[Double], vyDivide: IS[Double],
                         rotationDivide: IS[Double],
                         fuelDivide: IS[Double]){
  val dx = bound.width / xDivide
  val dy = bound.height / xDivide

  def getFeature(s: State): Feature = {
    def restrict(x: Int, bound: Int): Int = if(x < 0) 0 else if(x>=bound) bound - 1 else x

    def getIndex(dividers: IS[Double], v: Double): Int = {
      val i = dividers.indexWhere(d => v < d)
      if(i == -1) dividers.length else i
    }

    def encoding(bases: IS[Int], values: IS[Int]): Int = {
      var b = 1
      var acc = 0
      bases.indices.foreach{ i =>
        acc += values(i) * b
        b *= bases(i)
      }
      acc
    }

    val x = restrict(((s.pos.x + bound.width/2) / dx).toInt, xDivide)
    val y = restrict((s.pos.y / dy).toInt, yDivide)
    val vx = getIndex(vxDivide, s.velocity.x)
    val vy = getIndex(vyDivide, s.velocity.y)
    val rotation = getIndex(rotationDivide, s.rotation.angle)
    val fuel = getIndex(fuelDivide, s.fuelLeft)

    encoding(
      IS(xDivide, yDivide, vxDivide.length+1, vyDivide.length+1, rotationDivide.length+1, fuelDivide.length+1),
      IS(x,y,vx,vy,rotation,fuel))
  }

  def featureSpaceSize: Int = {
    IS(xDivide, yDivide, vxDivide.length+1, vyDivide.length+1, rotationDivide.length+1, fuelDivide.length+1).product
  }
}


class RocketLanding(world: spaceY.World,
                    terminateFunc: State => Option[spaceY.SimulationEnding],
                    rewardFunction: spaceY.RewardFunction,
                    initState: State,
                    actions: IS[Action],
                    featureExtract: RocketFeature) extends DynamicRLProblem[State, Feature, Action]{
  def sampleTransition(s: State, a: Action, random: Random): (State, Reward) = {
    val s1 = world.update(s, a)
    val r = terminateFunc(s1) match{
      case Some(ending) => rewardFunction.reward(ending)
      case None => 0.0
    }
    s1 -> r
  }

  def getFeature(s: State): Feature = {
    featureExtract.getFeature(s)
  }

  def sampleInitState(random: Random): State = {
    initState
  }

  def isTermState(s: State): Boolean = {
    terminateFunc(s).nonEmpty
  }

  def discountRate: Double = 1.0

  def availableActions(s: State): IS[Action] = actions
}


object RocketLanding{
  type Feature = Int


  def main(args: Array[String]): Unit = {
    import spaceY.Simulator._
    import spaceY.Geometry2D._

    val bound = WorldBound(width = 200, height = 150)
    val availableActions: IS[Action] = {
      for (
        rotate <- IS(1.0, 0.0, -1.0);
        thrust <- IS(0.1,0.2,0.4,0.8,1.0)
      ) yield Action(rotate, thrust)
    }

    val deltaT: Double = 1.0/20.0
    val world = World(gravity = Vec2.down*10, maxThrust = 20, deltaT = deltaT)

    val initState =
      State(pos = Vec2(20, 125), velocity = Vec2.zero, rotation = Rotation2(0), goalX = 75, fuelLeft = 10)


    val speedTolerance = math.sqrt(2.0 * world.gravity.magnitude * bound.height) / 2
    val taskParams = TaskParams(world, bound, availableActions, IS(initState),
      hitSpeedTolerance = speedTolerance, rotationTolerance = 0.40,
      rewardFunction = RewardFunction.LinearProduct(
        driftTolerance = bound.width / 3,
        rotationTolerance = 1.0 / 2,
        speedTolerance = speedTolerance,
        totalTime = 10.0))

    val terminateFunc = Simulator.standardTerminateFunc(taskParams.worldBound,
      taskParams.hitSpeedTolerance, taskParams.rotationTolerance) _

    def balanceThrustPolicy(state: State): (Action, PolicyInfo) = {
      val thrust = 0.4
      val rotSpeed = if (state.rotation.angle > 0) -1.0 else 1.0
      Action(rotationSpeed = rotSpeed, thrust) -> NoInfo
    }

    def sampleInitState(random: Random): State = {
      def between(from: Double, to: Double) = SimpleMath.linearInterpolate(from, to)(random.nextDouble())

      State(
        pos = Vec2(between(-80, 80), between(50, 125)),
        velocity = Vec2(between(-20, 20), between(-20, 20)),
        rotation = Rotation2(SimpleMath.cubic(between(-1.0, 1.0))),
        goalX = between(-20, 80),
        fuelLeft = 10
      )
    }


    val featureExtract = RocketFeature(bound, xDivide = 10, yDivide = 8,
      rotationDivide = IS(-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5),
      vxDivide = IS(-20.0, -10, -5, 0, 5, 10, 20),
      vyDivide = IS(10.0, 0.0, -5.0, -10.0, -20.0),
      fuelDivide = IS(0.1,0.3,0.6)
    )

    val problem = new RocketLanding(world, terminateFunc, taskParams.rewardFunction, initState,
      availableActions, featureExtract
    )

    val rand = new Random(1)
    val sarsa = new Sarsa(problem, exploreRate = t => 0.15 / (1 + t / 20000.0), learningRate = t => 0.5 , rand)
    val qMap: sarsa.QMap = mutable.HashMap()
    for(s <- 0 until featureExtract.featureSpaceSize; a <- availableActions){
      qMap(s -> a) = 0.0
    }

    println(s"Feature space size: ${featureExtract.featureSpaceSize}")

    def warmUpPolicy(state: State, time: Int): Action = {
      balanceThrustPolicy(state)._1
    }

    println("pre_train ...")
    (0 until 1000).foreach{ _ =>
      val s0 = sampleInitState(rand)
      sarsa.simulateAnEpisode(s0, warmUpPolicy(s0, 0), qMap, warmUpPolicy,
        sarsa.estimateState_expectedSarsa, 0)
    }
    println("pre_train finished")


    val traces = sarsa.runExpectedSarsa(qMap, startTime = 0).take(200000)

    var timeStep, episode = 0
    var scoreAcc = 0.0
    val scoreSmooth = 20.0

    var points: IS[(Double, Double)] = IS()
    var plot: Option[JFreeChart] = None

    val margin = 20

    val canvas = new JPanel(){
      setPreferredSize(new Dimension(600,400))

      override def paint(g: Graphics): Unit = {
        plot.foreach(_.draw(g.asInstanceOf[Graphics2D],
          new Rectangle2D.Double(margin, margin, getWidth-2*margin, getHeight-2*margin)))
      }
    }

    val frame = GUI.defaultFrame(canvas)


    traces.foreach{ trace =>
      val score = trace.last._3
      println(s"length: ${trace.length}, reward: ${score}")

      scoreAcc = scoreAcc * (1.0-1.0/scoreSmooth) + 1.0/scoreSmooth * score

      points :+= (episode.toDouble -> scoreAcc)

      if(episode % 200 == 0){
        plot = Some(ListPlot.plot("performance" -> points)("Performance Curve",
          xLabel = "Episode", yLabel = "Score"
        ))
        canvas.repaint()
      }

      timeStep += trace.length
      episode += 1
    }
  }
}