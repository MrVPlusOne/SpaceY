package spaceY

import java.awt.Dimension

import javax.swing.JFrame
import org.apache.log4j.BasicConfigurator
import org.nd4j.linalg.factory.Nd4j
import rx._
import spaceY.Geometry2D._
import spaceY.Simulator.{FullSimulation, NoInfo, PolicyInfo, WorldBound}

import scala.util.Random

object Playground {

  val deltaT: Double = 1.0/25.0
  val world = World(gravity = Vec2.down*10, maxThrust = 30, deltaT = deltaT)

  def main(args: Array[String]): Unit = {
    testUI(args)
    return

//    BasicConfigurator.configure()

    val bound = WorldBound(width = 200, height = 150)
    val train = new Training(world, bound,
      initFuel = 10, hitSpeedTolerance = 12, rotationTolerance = 0.4 * math.Pi,
      initState = State(pos = Vec2(-50, 125), velocity = Vec2.zero, rotation = 0, goalX = 20, fuelLeft = 10)
    )

    train.train(1000, netReplaceRate = 0.01,
      exploreRateFunc = e => 1.0/(5.0+e.toDouble/100))
  }

  def testUI(args: Array[String]): Unit = {
    val bound = WorldBound(width = 250, height = 150)
    val initState = State(Vec2(50, 100), Vec2.left * 10, math.Pi * 0.5, goalX = 10, fuelLeft = 10)

    def rotatePolicy(state: State): Action = {
      Action(rotationSpeed = 0.3 * math.Pi, thrust = 0.9)
    }

    val rand = new Random(1)
    def randomPolicy(state: State): (Action, PolicyInfo) = {
      Action(rotationSpeed = (2*rand.nextDouble()-1) * math.Pi, thrust = rand.nextDouble()) -> NoInfo
    }

    val simResult = Simulator(initState,
      world,
      terminateFunc = Simulator.standardTerminateFunc(bound, hitSpeedTolerance = 5, rotationTolerance = 0.15 * math.Pi)
    ).simulateUntilResult(randomPolicy)

    println(s"Ending: ${simResult.ending}")


    Rx.unsafe {

      val scPanel = new StateWithControlPanel(bound, IS(simResult), 0, 0)


      scPanel.jPanel.setPreferredSize(new Dimension(800, 500))

      val frame = new JFrame()
      frame.setContentPane(scPanel.jPanel)
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.pack()
      frame.setVisible(true)
    }
  }
}
