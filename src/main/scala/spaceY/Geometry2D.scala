package spaceY

object Geometry2D {

  case class Vec2(x: Double, y: Double) {

    def magnitude: Double = math.sqrt(x * x + y * y)

    def +(v: Vec2): Vec2 = Vec2(this.x + v.x, this.y + v.y)

    def -(v: Vec2): Vec2 = Vec2(this.x - v.x, this.y - v.y)

    def *(d: Double): Vec2 = Vec2(d * x, d * y)

    def rotate(angle: Double): Vec2 = {
      val c = math.cos(angle)
      val s = math.sin(angle)
      Vec2(x * c - s * y, y * c + s * x)
    }

    override def toString: String = {
      "(%.2f, %.2f)".format(x,y)
    }
  }

  object Vec2 {
    val up: Vec2 = Vec2(0, 1)
    val down: Vec2 = Vec2(0, -1)
    val left: Vec2 = Vec2(-1, 0)
    val right: Vec2 = Vec2(1, 0)
    val zero = Vec2(0, 0)
  }

  case class Line2(from: Vec2, to: Vec2)

}
