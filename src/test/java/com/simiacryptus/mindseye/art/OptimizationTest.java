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

package com.simiacryptus.mindseye.art;

import com.simiacryptus.mindseye.art.models.Inception5H;
import com.simiacryptus.mindseye.art.ops.ChannelMeanMatcher;
import com.simiacryptus.mindseye.art.ops.GramMatrixMatcher;
import com.simiacryptus.mindseye.art.ops.ChannelMeanEnhancer;
import com.simiacryptus.mindseye.art.ops.ContentMatcher;
import com.simiacryptus.mindseye.art.util.VisionPipelineUtil;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.BisectionSearch;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.TestUtil;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OptimizationTest {
  private static final Logger log = LoggerFactory.getLogger(OptimizationTest.class);

  public static void train(Tensor image, PipelineNetwork network, int maxIterations, LineSearchStrategy lineSearch) {
    TestUtil.monitorImage(image, false, 5, false);
    new IterativeTrainer(new ArrayTrainable(new Tensor[][]{{image}}, MultiPrecision.setPrecision(network, Precision.Float)).setMask(true))
        .setOrientation(new TrustRegionStrategy(new GradientDescent()) {
          @Override
          public TrustRegion getRegionPolicy(final Layer layer1) {
            return new RangeConstraint().setMin(0e-2).setMax(256);
          }
        })
        .setMonitor(getTrainingMonitor())
        .setMaxIterations(maxIterations)
        .setLineSearchFactory(name -> lineSearch)
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
  }

  @NotNull
  public static TrainingMonitor getTrainingMonitor() {
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(String msg) {
        System.out.println(msg);
        super.log(msg);
      }

      @Override
      public void onStepComplete(Step currentPoint) {
        super.onStepComplete(currentPoint);
      }
    };
  }

  @Test
  public void testDream() throws InterruptedException {
    Tensor image = Tensor.fromRGB(VisionPipelineUtil.load("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg", 500));
    train(image, new ChannelMeanEnhancer().build(Inception5H.Inc5H_3b, image), 100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1));
    Thread.sleep(100000);
  }

  @Test
  public void testStyleTransfer() throws InterruptedException {
    Tensor contentImage = Tensor.fromRGB(VisionPipelineUtil.load("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg", 500));
    Tensor styleImage = Tensor.fromRGB(VisionPipelineUtil.load("https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200));
    train(contentImage, (PipelineNetwork) MultiPrecision.setPrecision(SumInputsLayer.combine
        (
            new GramMatrixMatcher().build(Inception5H.Inc5H_1a, styleImage),
            new GramMatrixMatcher().build(Inception5H.Inc5H_2a, styleImage),
            new GramMatrixMatcher().build(Inception5H.Inc5H_3a, styleImage),
            new ContentMatcher().build(contentImage)
            //.andThenWrap(new LinearActivationLayer().setScale(1e0).freeze())
        ), Precision.Float), 100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-4)
    );
    Thread.sleep(100000);
  }

  @Test
  public void testMeanMatch() throws InterruptedException {
    train(
        Tensor.fromRGB(VisionPipelineUtil.load("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg", 500)),
        new ChannelMeanMatcher().build(Inception5H.Inc5H_1a,
            Tensor.fromRGB(VisionPipelineUtil.load("https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200))),
        100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1)
    );
    Thread.sleep(100000);
  }

  @Test
  public void testGramMatch() throws InterruptedException {
    train(
        Tensor.fromRGB(VisionPipelineUtil.load("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg", 500)),
        new GramMatrixMatcher().build(Inception5H.Inc5H_2a,
            Tensor.fromRGB(VisionPipelineUtil.load("https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200))),
        100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1)
    );
    Thread.sleep(100000);
  }

}
