package spaceY

import java.awt.{Component, Container, Dimension}

import javax.swing._

object GUI {
  def panel(horizontal: Boolean)(children: Component*): JPanel = {
    val p = new JPanel()
    p.setLayout(new BoxLayout(p, if(horizontal) BoxLayout.X_AXIS else BoxLayout.Y_AXIS))
    children.foreach(c => p.add(c))
    p
  }

  def defaultFrame(content: Container, width: Int = 600, height: Int = 400) = {
    new JFrame(){
      setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
      setContentPane(content)
      pack()
      setVisible(true)
    }
  }

  def sliderLabel(labelSize: Dimension, slider: JSlider): JLabel = {
    def calcContent(): String ={
      slider.getValue.toString
    }

    val label = new JLabel(calcContent()){
      setMinimumSize(labelSize)
      setMaximumSize(labelSize)
      setPreferredSize(labelSize)
    }
    slider.addChangeListener{_ =>
      label.setText(calcContent())
    }
    label
  }
}
