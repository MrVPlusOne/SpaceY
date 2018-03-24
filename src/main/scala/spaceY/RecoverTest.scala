package spaceY

object RecoverTest {
  def main(args: Array[String]): Unit = {
    val visualizer = TraceVisualizer.recoverFromData("/Users/weijiayi/Programming/RL/SpaceY/results/18-03-22-14.52.45[ioId=32]/visualizerData.serialized")

    visualizer.initializeFrame()
  }
}
