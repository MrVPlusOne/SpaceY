package spaceY

object Geometry2D {

  case class Vec2(x: Double, y: Double) {

    def magnitude: Double = math.sqrt(x * x + y * y)

    def +(v: Vec2): Vec2 = Vec2(this.x + v.x, this.y + v.y)

    def -(v: Vec2): Vec2 = Vec2(this.x - v.x, this.y - v.y)

    def *(d: Double): Vec2 = Vec2(d * x, d * y)

    def rotate(rot: Rotation2): Vec2 = {
      val c = rot.cos
      val s = rot.sin
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

  @SerialVersionUID(0)
  class Rotation2 private(val angle: Double) extends Serializable {
    override def equals(obj: scala.Any): Boolean = obj match {
      case r: Rotation2 => angle == r.angle
    }

    override def toString: String = "r%.2f".format(angle)

    def sin: Double = math.sin(angle * math.Pi)

    def cos: Double = math.cos(angle * math.Pi)

    def rotate(delta: Double): Rotation2 = Rotation2(angle + delta)

    def abs: Double = math.abs(angle)
  }

  object Rotation2 {
    def apply(angle: Double): Rotation2 = {
      new Rotation2(SimpleMath.wrapInRange(angle, -1.0, 1.0))
    }
  }

}
