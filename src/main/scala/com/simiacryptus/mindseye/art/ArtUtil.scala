/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art

import java.io.File

import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.network.{DAGNetwork, DAGNode, PipelineNetwork}
import com.simiacryptus.mindseye.opt.{Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.{StepRecord, TestUtil}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.util.Util
import guru.nidi.graphviz.engine.{Format, Graphviz}
import guru.nidi.graphviz.model.Graph

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

object ArtUtil {

  def withTrainingMonitor[T](log: NotebookOutput, fn: TrainingMonitor => T) = {
    val history = new ArrayBuffer[StepRecord]
    NotebookRunner.withMonitoredImage(log, Util.toImage(TestUtil.plot(history))) {
      val trainingMonitor = new TrainingMonitor() {
        override def clear(): Unit = {
          super.clear()
        }

        override def log(msg: String): Unit = {
          System.out.println(msg)
          super.log(msg)
        }

        override def onStepComplete(currentPoint: Step): Unit = {
          history += new StepRecord(currentPoint.point.getMean, currentPoint.time, currentPoint.iteration)
          super.onStepComplete(currentPoint)
        }
      }
      fn(trainingMonitor)
    }
  }

  def graph(log: NotebookOutput, network: PipelineNetwork) = {
    val graphviz = Graphviz.fromGraph(TestUtil.toGraph(network, Java8Util.cvt((node: DAGNode) => {
      Option(node.getLayer[Layer]()).map(_.getName.stripSuffix("Layer")).getOrElse("Input " + node.getNetwork.inputHandles.indexOf(node.getId))
    })).asInstanceOf[Graph])
    log.p(log.png(graphviz.height(400).width(600).render(Format.PNG).toImage, "Configuration Graph"))
    log.p(log.svg(graphviz.height(400).width(600).render(Format.SVG_STANDALONE).toString, "Configuration Graph"))
  }


}
