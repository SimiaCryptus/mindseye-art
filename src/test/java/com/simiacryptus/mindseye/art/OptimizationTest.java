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
import com.simiacryptus.mindseye.art.ops.ChannelPowerEnhancer;
import com.simiacryptus.mindseye.art.ops.ContentMatcher;
import com.simiacryptus.mindseye.art.ops.GramMatrixMatcher;
import com.simiacryptus.mindseye.art.util.ImageArtUtil;
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
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NullNotebookOutput;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

/**
 * The type Optimization test.
 */
public class OptimizationTest {
  private static final Logger log = LoggerFactory.getLogger(OptimizationTest.class);

  /**
   * Gets training monitor.
   *
   * @return the training monitor
   */
  @Nonnull
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

  /**
   * Train.
   *
   * @param image         the image
   * @param network       the network
   * @param maxIterations the max iterations
   * @param lineSearch    the line search
   */
  public static void train(Tensor image, PipelineNetwork network, int maxIterations, LineSearchStrategy lineSearch) {
    ImageUtil.monitorImage(image.addRef(), false, 5, false);
    MultiPrecision.setPrecision(network.addRef(), Precision.Float);
    ArrayTrainable arrayTrainable = new ArrayTrainable(new Tensor[][]{{image}}, network);
    arrayTrainable.setMask(true);
    IterativeTrainer iterativeTrainer = new IterativeTrainer(arrayTrainable);
    iterativeTrainer.setOrientation(new TrustRegionStrategy(new GradientDescent()) {
      @Nonnull
      @Override
      public TrustRegion getRegionPolicy(final Layer layer1) {
        if (null != layer1) layer1.freeRef();
        return new RangeConstraint().setMin(0e-2).setMax(256);
      }

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
      }
    });
    iterativeTrainer.setMonitor(getTrainingMonitor());
    iterativeTrainer.setMaxIterations(maxIterations);
    iterativeTrainer.setLineSearchFactory(name -> lineSearch);
    iterativeTrainer.setTerminateThreshold(Double.NEGATIVE_INFINITY);
    iterativeTrainer.run();
    iterativeTrainer.freeRef();
  }

  /**
   * Test dream.
   *
   * @throws InterruptedException the interrupted exception
   */
  @Test
  public void testDream() throws InterruptedException {
    Tensor image = Tensor.fromRGB(ImageArtUtil.loadImage(new NullNotebookOutput(),
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg",
        500));
    train(image, new ChannelPowerEnhancer().build(Inception5H.Inc5H_3b, null, null, image.addRef()), 100,
        new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1));
    Thread.sleep(100000);
  }

  /**
   * Test style transfer.
   *
   * @throws InterruptedException the interrupted exception
   */
  @Test
  public void testStyleTransfer() throws InterruptedException {
    NullNotebookOutput log = new NullNotebookOutput();
    Tensor contentImage = Tensor.fromRGB(ImageArtUtil.loadImage(log,
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg",
        500));
    Tensor styleImage = Tensor.fromRGB(ImageArtUtil.loadImage(log,
        "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200));
    PipelineNetwork pipelineNetwork = SumInputsLayer.combine(new GramMatrixMatcher().build(Inception5H.Inc5H_1a, null, null, styleImage.addRef()),
        new GramMatrixMatcher().build(Inception5H.Inc5H_2a, null, null, styleImage.addRef()),
        new GramMatrixMatcher().build(Inception5H.Inc5H_3a, null, null, styleImage.addRef()),
        new ContentMatcher().build(VisionPipelineLayer.NOOP, null, null, contentImage.addRef())
        //.andThenWrap(new LinearActivationLayer().setScale(1e0).freeze())
    );
    styleImage.freeRef();
    MultiPrecision.setPrecision(pipelineNetwork.addRef(), Precision.Float);
    train(contentImage, pipelineNetwork, 100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-4));
    Thread.sleep(100000);
  }

  /**
   * Test mean match.
   *
   * @throws InterruptedException the interrupted exception
   */
  @Test
  public void testMeanMatch() throws InterruptedException {
    NullNotebookOutput log = new NullNotebookOutput();
    train(Tensor.fromRGB(ImageArtUtil.loadImage(log,
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg",
        500)),
        new ChannelMeanMatcher().build(Inception5H.Inc5H_1a, null, null,
            Tensor.fromRGB(ImageArtUtil.loadImage(log,
                "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200))),
        100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1));
    Thread.sleep(100000);
  }

  /**
   * Test gram match.
   *
   * @throws InterruptedException the interrupted exception
   */
  @Test
  public void testGramMatch() throws InterruptedException {
    NullNotebookOutput log = new NullNotebookOutput();
    train(Tensor.fromRGB(ImageArtUtil.loadImage(log,
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg",
        500)),
        new GramMatrixMatcher().build(Inception5H.Inc5H_2a, null, null,
            Tensor.fromRGB(ImageArtUtil.loadImage(log,
                "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200))),
        100, new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1));
    Thread.sleep(100000);
  }

}
