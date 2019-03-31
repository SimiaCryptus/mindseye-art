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

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.ArtUtil._
import com.simiacryptus.mindseye.art.constraints.{GramMatrixMatcher, RMSContentMatcher}
import com.simiacryptus.mindseye.art.models.InceptionVision
import com.simiacryptus.mindseye.eval.ArrayTrainable
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{CoreSettings, Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.BisectionSearch
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{RangeConstraint, TrustRegion}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, InteractiveSetup}

object SimpleStyleTransfer_EC2 extends SimpleStyleTransfer() with AWSNotebookRunner[Object] with EC2Runner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap: Option[String] = Option("55g")

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P2_XL

//  override def javaProperties: Map[String, String] = Map(
//    "spark.master" -> "local[8]",
//    "MAX_TOTAL_MEMORY" -> (8 * CudaMemory.GiB).toString,
//    "MAX_DEVICE_MEMORY" -> (8 * CudaMemory.GiB).toString,
//    "MAX_IO_ELEMENTS" -> (2 * CudaMemory.MiB).toString,
//    "CONVOLUTION_WORKSPACE_SIZE_LIMIT" -> (1 * 512 * CudaMemory.MiB).toString,
//    "MAX_FILTER_ELEMENTS" -> (1 * 512 * CudaMemory.MiB).toString,
//    "java.util.concurrent.ForkJoinPool.common.parallelism" -> Integer.toString(CoreSettings.INSTANCE().jvmThreads)
//  )

}

abstract class SimpleStyleTransfer extends InteractiveSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 600
  val styleResolution = 1280
  val trainingMinutes: Int = 20
  val trainingIterations: Int = 100
  val isVerbose: Boolean = false

  def precision: Precision = Precision.Float

  override def postConfigure(log: NotebookOutput) = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(10000)
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))
    val styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))
    withMonitoredImage(log, contentImage.toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.run(() => {
          val gramMatcher = new GramMatrixMatcher()

          val contentMatcher = new RMSContentMatcher()

          val network = MultiPrecision.setPrecision(SumInputsLayer.combine(
            gramMatcher.build(InceptionVision.Layer1a.getNetwork, styleImage),
            gramMatcher.build(InceptionVision.Layer2a.getNetwork, styleImage),
            gramMatcher.build(InceptionVision.Layer3a.getNetwork, styleImage),
            gramMatcher.build(InceptionVision.Layer3b.getNetwork, styleImage),
            contentMatcher.build(new PipelineNetwork, contentImage)
          ), precision).asInstanceOf[PipelineNetwork]

          val lineSearch = new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-4)

          new IterativeTrainer(new ArrayTrainable(Array[Array[Tensor]](Array(contentImage)), network).setMask(true))
            .setOrientation(new TrustRegionStrategy(new GradientDescent) {
              override def getRegionPolicy(layer1: Layer): TrustRegion = new RangeConstraint().setMin(0e-2).setMax(256)
            })
            .setMonitor(trainingMonitor)
            .setMaxIterations(trainingIterations)
            .setLineSearchFactory((_:CharSequence) => lineSearch)
            .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
            .runAndFree
        }: Unit)
        null
      })
    }
  }

}
