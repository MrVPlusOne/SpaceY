package spaceY

import java.awt._
import java.awt.geom.{GeneralPath, Line2D}

import javax.swing._
import rx._
import spaceY.Geometry2D._
import spaceY.Simulator.{FullSimulation, NoInfo, PolicyInfo, WorldBound}

case class Visualization(worldBound: WorldBound, state: State, action: Action, info: PolicyInfo) {
  val borderThickness = 5
  val rocketHeight = worldBound.height / 12
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

    def drawInfo() = {
      val offset = vecTrans(Vec2(-worldBound.width/2, worldBound.height))
      val dx = offset.x.toInt
      val dy = offset.y.toInt
      g2D.setColor(Color.green.darker())
      g2D.drawString(info.displayInfo, dx + 10 + borderThickness, dy + 20 + borderThickness)
      g2D.drawString(action.toString, dx + 10 + borderThickness, dy + 40 + borderThickness)
    }

    def drawGoal() = {
      g2D.setStroke(new BasicStroke(borderThickness, BasicStroke.CAP_BUTT,
        BasicStroke.JOIN_MITER, 10f, Array(10f), 0f))
      g2D.setColor(Color.green.brighter())
      val goalLine = Line2(Vec2(state.goalX, h), Vec2(state.goalX, 0))
      drawLine(goalLine)
    }

    def drawBorder() = {
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
    }
    def drawRocket() = {
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


    drawGoal()
    drawBorder()
    drawRocket()
    drawInfo()
  }
}

class TraceVisualizer(private var traces: IS[(String, FullSimulation)],
                      private var scores: IS[Double],
                      worldBound: WorldBound) {
  def dataNum = traces.length

  def addTrace(name: String, simulation: FullSimulation, score: Double) = {
    traces :+= (name, simulation)
    scores :+= score
  }

  val placeholder = GUI.panel(horizontal = false)()
  val dataButton = new JButton("Fetch data")
  val frame = new JFrame()
  frame.setContentPane(
    GUI.panel(horizontal = false)(
      placeholder,
      GUI.panel(horizontal = true)(dataButton, GUI.panel(horizontal = false)())
    )
  )
  var oldSCPanel: Option[StateWithControlPanel] = None

  def initializeFrame(): Unit = {
    require(traces.nonEmpty)
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.setVisible(true)
    redisplayData()
  }

  def redisplayData(): Unit = {
    import rx.Ctx.Owner.Unsafe._

    val newP = oldSCPanel match {
      case None =>
        new StateWithControlPanel(worldBound, traces, scores, 0, 0) {
          jPanel.setPreferredSize(new Dimension(600, ((worldBound.height / worldBound.width) * 600).toInt))
        }
      case Some(op) =>
        val np = new StateWithControlPanel(worldBound, traces, scores, op.simulation, op.step)
        np.jPanel.setPreferredSize(op.jPanel.getSize)
        op.stopTracking()
        np
    }

    placeholder.removeAll()
    placeholder.add(newP.jPanel)
    frame.pack()

    oldSCPanel = Some(newP)
  }


  dataButton.addActionListener(_ => {
    redisplayData()
  })
}

class StatePanel(visual: Var[Visualization])(implicit ctx: Ctx.Owner){
  val margin = 10

  val jPanel: JPanel = new JPanel(){
    override def paintComponent(g: Graphics): Unit = {
      val g2D = g.asInstanceOf[Graphics2D]
      g2D.translate(margin, margin)
      g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
      visual.now.draw(g.asInstanceOf[Graphics2D], getWidth - 2*margin, getHeight - 2*margin)
    }
  }

  private val repaintObs = visual.trigger{
    jPanel.repaint()
  }

  def stopTracking(): Unit = repaintObs.kill()
}

class StateWithControlPanel(worldBound: WorldBound, simulations: IS[(String, FullSimulation)], scores: IS[Double],
                            var simulation: Int, var step: Int)(implicit ctx: Ctx.Owner){
  val sliderSize = new Dimension(200,20)


  def mkVisual() = {
    val (s, a) = simulations(simulation)._2.trace(step)
    Visualization(worldBound, s, a._1, a._2)
  }

  val visual = Var {
    mkVisual()
  }

  val resultArea = new JLabel()
  val nameLabel = new JLabel()
  def setResult(): Unit ={
    val (name, fullSim) = simulations(simulation)
    nameLabel.setText(s"Name: $name")
    resultArea.setText(s"Score: ${scores(simulation)}, ${fullSim.ending}")
  }
  setResult()

  val simulationSelector: JSlider = new JSlider(0, simulations.length-1, simulation){
    setPreferredSize(sliderSize)
    setMajorTickSpacing(1)
    addChangeListener{_ =>
      simulation = getValue
      setResult()
      resetStepSelector(simulation)
    }
  }
  val stepSelector: JSlider = new JSlider(0, simulations(simulation)._2.trace.length-1, step){
    setPreferredSize(sliderSize)
    setMajorTickSpacing(1)
  }


  def resetStepSelector(simulation: Int): Unit ={
    stepSelector.setMaximum(simulations(simulation)._2.trace.length-1)
    stepSelector.setValue(0)
  }
  stepSelector.addChangeListener{_ =>
    step = stepSelector.getValue
    visual() = mkVisual()
  }

  val statePanel = new StatePanel(visual)

  import GUI._


  val jPanel: JPanel = {
    val labelDim = new JLabel("1234").getPreferredSize

    panel(horizontal = false)(
      statePanel.jPanel,
      nameLabel,
      resultArea,
      panel(horizontal = true)(
        panel(horizontal = true)(
          new JLabel("simulation: "),
          sliderLabel(labelDim, simulationSelector),
          simulationSelector),
        panel(horizontal = true)(
          new JLabel("step: "),
          sliderLabel(labelDim, stepSelector) , stepSelector)
      )
    )
  }

  def stopTracking(): Unit = statePanel.stopTracking()
}

object TestVisual{
  def main(args: Array[String]): Unit = {
    Rx.unsafe {
      val visual = Var {
        val s = State(Vec2(1.3, 1), Vec2.zero, Rotation2(0.2), goalX = 0.5, fuelLeft = 3)
        Visualization(WorldBound(width = 600, height = 500), s, Action(0,0), NoInfo)
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