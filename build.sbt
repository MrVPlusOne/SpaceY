name := "SpaceY"

version := "0.1"

scalaVersion := "2.12.4"

resolvers += Resolver.sonatypeRepo("snapshots")

libraryDependencies += "com.lihaoyi" %% "scalarx" % "0.3.2"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.jfree" % "jfreechart" % "1.0.14",
  "org.nd4j" % "nd4j-native-platform" % "0.9.1",
//  "org.nd4j" % "nd4j-cuda-9.1" % "0.9.2-SNAPSHOT",
  "org.slf4j" % "slf4j-api" % "1.7.5",
  "org.slf4j" % "slf4j-log4j12" % "1.7.5",
  "com.typesafe.akka" %% "akka-actor" % "2.5.6",
  "com.typesafe.akka" %% "akka-testkit" % "2.5.6" % Test,

  "com.lihaoyi" %% "ammonite-ops" % "1.0.3"
)

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)


resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

//libraryDependencies += "org.platanios" %% "tensorflow" % "0.1.1" classifier "darwin-cpu-x86_64"
//libraryDependencies += "org.platanios" %% "tensorflow" % "0.1.1"