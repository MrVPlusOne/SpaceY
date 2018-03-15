package spaceY

import java.awt.Dimension
import javax.swing.JFrame

import rx._
import spaceY.Geometry2D._
import spaceY.Simulator.WorldBound

import scala.util.Random

object Playground {

  val deltaT: Double = 1.0/25.0
  val world = World(gravity = Vec2.down*10, maxThrust = 30, deltaT = deltaT)

  def main(args: Array[String]): Unit = {
    val bound = WorldBound(width = 200, height = 150)
    val train = new Training(world, bound, initFuel = 20, hitSpeedTolerance = 12, rotationTolerance = 0.4 * math.Pi)
    train.trainPolicy()
  }

  def testUI(args: Array[String]): Unit = {
    val bound = WorldBound(width = 250, height = 150)
    val initState = State(Vec2(50, 100), Vec2.left * 10, math.Pi * 0.5, goalX = 10, fuelLeft = 2)

    def rotatePolicy(state: State): Action = {
      Action(rotationSpeed = 0.3 * math.Pi, thrust = 0.9)
    }

    val rand = new Random(1)
    def randomPolicy(state: State): Action = {
      Action(rotationSpeed = (2*rand.nextDouble()-1) * math.Pi, thrust = rand.nextDouble())
    }

    val (states, ending) = Simulator(initState,
      world,
      terminateFunc = Simulator.standardTerminateFunc(bound, hitSpeedTolerance = 5, rotationTolerance = 0.15 * math.Pi)
    ).simulateUntilResult(randomPolicy)

    println(s"Ending: $ending")


    Rx.unsafe {

      val visual = Var {
        val (s,a) = states.head
        Visualization(bound, s, a)
      }

      val animation = new Thread(() => {
        states.tail.foreach{ case (s,a) =>
          visual() = Visualization(bound, s, a)
          Thread.sleep((deltaT * 1000).toInt)
        }
      })


      val statePanel = new StatePanel(visual)
      statePanel.jPanel.setPreferredSize(new Dimension(800, 500))

      val frame = new JFrame()
      frame.setContentPane(statePanel.jPanel)
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.pack()
      frame.setVisible(true)

      animation.start()
    }
  }
}
