package spaceY

import java.awt._
import java.awt.geom.{GeneralPath, Line2D}
import javax.swing.{JComponent, JFrame, JPanel}

import rx._
import spaceY.Geometry2D._
import spaceY.Simulator.WorldBound

case class Visualization(worldBound: WorldBound, state: State, action: Action) {
  val borderThickness = 5
  val rocketHeight = worldBound.height / 15
  val rocketHalfWidth = rocketHeight/6

  def draw(g2D: Graphics2D, boxW: Double, boxH: Double): Unit = {
    val hScale = boxH / worldBound.height
    val wScale = boxW / worldBound.width
    val scale = math.min(hScale, wScale)

    def vecTrans(v: Vec2): Vec2 = {
      val y = boxH - v.y * scale
      val x = boxW / 2 + v.x * scale
      Vec2(x, y)
    }

    def drawLine(line2D: Line2): Unit = {
      val from = vecTrans(line2D.from)
      val to = vecTrans(line2D.to)
      g2D.draw(new Line2D.Double(from.x, from.y, to.x, to.y))
    }

    def makeClosePath(ps: Seq[Vec2]): GeneralPath = {
      val path = new GeneralPath()
      val points = ps.map(vecTrans)
      path.moveTo(points.head.x, points.head.y)
      points.tail.foreach { p => path.lineTo(p.x, p.y) }
      path.closePath()
      path
    }

    val w = worldBound.width
    val h = worldBound.height


    //draw goal
    g2D.setStroke(new BasicStroke(borderThickness, BasicStroke.CAP_BUTT,
      BasicStroke.JOIN_MITER, 10f, Array(10f), 0f))
    g2D.setColor(Color.green.brighter())
    val goalLine = Line2(Vec2(state.goalX, h), Vec2(state.goalX, 0))
    drawLine(goalLine)


    //draw border
    g2D.setStroke(new BasicStroke(borderThickness))
    val groundLine = Line2(Vec2(-w / 2, 0), Vec2(w / 2, 0))
    val leftLine = Line2(Vec2(-w / 2, h), Vec2(-w / 2, 0))
    val rightLine = Line2(Vec2(w / 2, h), Vec2(w / 2, 0))
    val topLine = Line2(Vec2(-w / 2, h), Vec2(w / 2, h))
    g2D.setColor(Color.red)
    drawLine(leftLine)
    drawLine(rightLine)
    drawLine(topLine)
    g2D.setColor(Color.black)
    drawLine(groundLine)

    //draw flare
    g2D.setPaint(Color.red)
    val bottom = state.pos - Vec2.up.rotate(state.rotation) * (rocketHeight / 3)
    val flareScale = action.thrust
    val flareLeft = bottom + Vec2.left.rotate(state.rotation) * (rocketHalfWidth * 0.6 * flareScale)
    val flareRight = bottom + Vec2.right.rotate(state.rotation) * (rocketHalfWidth * 0.6 * flareScale)
    val flareBottom = bottom + Vec2.down.rotate(state.rotation) * (rocketHeight * 0.8 * flareScale)
    g2D.fill(makeClosePath(Seq(flareLeft, flareBottom, flareRight)))


    //draw rocket
    g2D.setPaint(Color.blue)
    val tip = bottom + Vec2.up.rotate(state.rotation) * rocketHeight
    val rocketLeft = bottom + Vec2.left.rotate(state.rotation) * rocketHalfWidth
    val rocketRight = bottom + Vec2.right.rotate(state.rotation) * rocketHalfWidth
    g2D.fill(makeClosePath(Seq(tip, rocketLeft, rocketRight)))
  }
}

class StatePanel(visual: Var[Visualization])(implicit ctx: Ctx.Owner){
  val margin = 10

  val jPanel: JPanel = new JPanel(){
    override def paintComponent(g: Graphics): Unit = {
      g.asInstanceOf[Graphics2D].translate(margin, margin)
      visual.now.draw(g.asInstanceOf[Graphics2D], getWidth - 2*margin, getHeight - 2*margin)
    }
  }

  visual.trigger{
    jPanel.repaint()
  }
}

object TestVisual{
  def main(args: Array[String]): Unit = {
    Rx.unsafe {
      val visual = Var {
        val s = State(Vec2(1.3, 1), Vec2.zero, math.Pi * 0.2, goalX = 0.5, fuelLeft = 3)
        Visualization(WorldBound(width = 600, height = 500), s, Action(0,0))
      }

      val statePanel = new StatePanel(visual)
      statePanel.jPanel.setPreferredSize(new Dimension(500, 400))

      val frame = new JFrame()
      frame.setContentPane(statePanel.jPanel)
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.pack()
      frame.setVisible(true)
    }
  }
}