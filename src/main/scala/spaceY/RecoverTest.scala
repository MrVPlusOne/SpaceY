package spaceY

object RecoverTest {
  def main(args: Array[String]): Unit = {
    val visualizer = TraceVisualizer.recoverFromData("/Users/weijiayi/Programming/RL/SpaceY/results/18-03-20-14:22:27/visualizerData.serialized")

    visualizer.initializeFrame()
  }
}
