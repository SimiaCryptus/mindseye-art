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
package com.simiacryptus.mindseye.art.models;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgReshapeLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;

class VGG16_HDF5 {

  public static final Logger log = LoggerFactory.getLogger(VGG16_HDF5.class);
  protected static final Logger logger = LoggerFactory.getLogger(VGG16_HDF5.class);
  public final Hdf5Archive hdf5;
  @Nonnull
  int[] convolutionOrder = {3, 2, 0, 1};
  @Nonnull
  int[] fullyconnectedOrder = {1, 0};
  private PoolingLayer.PoolingMode finalPoolingMode = PoolingLayer.PoolingMode.Max;
  private boolean large = true;
  private boolean dense = true;

  public VGG16_HDF5(Hdf5Archive hdf5) {
    this.hdf5 = hdf5;
  }

  public static VGG16_HDF5 fromHDF5() {
    try {
      return fromHDF5(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")));
    } catch (IOException | KeyManagementException | NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }

  public static VGG16_HDF5 fromHDF5(final File hdf) {
    try {
      return new VGG16_HDF5(new Hdf5Archive(hdf));
    } catch (@Nonnull final RuntimeException e) {
      throw e;
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  protected static Layer add(@Nonnull Layer layer, @Nonnull PipelineNetwork model) {
    if (layer instanceof Explodable) {
      Layer explode = ((Explodable) layer).explode();
      try {
        if (explode instanceof DAGNetwork) {
          logger.info(String.format(
              "Exploded %s to %s (%s nodes)",
              layer.getName(),
              explode.getClass().getSimpleName(),
              ((DAGNetwork) explode).getNodes().size()
          ));
        } else {
          logger.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), explode.getName()));
        }
        return add(explode, model);
      } finally {
        layer.freeRef();
      }
    } else {
      model.wrap(layer).freeRef();
      return layer;
    }
  }

  public Layer buildNetwork() {
    PipelineNetwork pipeline = new PipelineNetwork();
    phase0(pipeline);
    phase1(pipeline);
    phase2(pipeline);
    phase3(pipeline);
    return pipeline;
  }

  public void phase1(PipelineNetwork pipeline) {
    phase1a(pipeline);
    phase1b(pipeline);
    phase1c(pipeline);
    phase1d(pipeline);
    phase1e(pipeline);
  }

  public void phase0(PipelineNetwork pipeline) {
    add(new ImgMinSizeLayer(226, 226), pipeline);
    add(new ImgBandBiasLayer(3).setAndFree(new Tensor(-103.939, -116.779, -123.68)), pipeline);
  }

  public void phase1a(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1", pipeline);
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3", pipeline);
  }

  public void phase1b(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 64, 128, ActivationLayer.Mode.RELU, "layer_6", pipeline);
    addConvolutionLayer(3, 128, 128, ActivationLayer.Mode.RELU, "layer_8", pipeline);
  }

  public void phase1c(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11", pipeline);
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13", pipeline);
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15", pipeline);
  }

  public void phase1d(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_18", pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_20", pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22", pipeline);
  }

  public void phase1e(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_25", pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_27", pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29", pipeline);
  }

  public void phase2(PipelineNetwork pipeline) {
    phase2a(pipeline);
    phase2b(pipeline);
  }

  public void phase2a(PipelineNetwork pipeline) {
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    addPoolingLayer(2, pipeline);
  }

  public void phase2b(PipelineNetwork pipeline) {
    if (large) {
      add(new ImgModulusPaddingLayer(7, 7), pipeline);
    } else {
      add(new ImgModulusPaddingLayer(-7, -7), pipeline);
    }

    if (dense) {
      add(new ConvolutionLayer(7, 7, 512, 4096)
          .setStrideXY(1, 1)
          .setPaddingXY(0, 0)
          .setAndFree(hdf5.readDataSet("param_0", "layer_32")
              .reshapeCastAndFree(7, 7, 512, 4096).permuteDimensionsAndFree(0, 1, 3, 2)
          ), pipeline);
    } else {
      add(new ImgModulusPaddingLayer(7, 7), pipeline);
      add(new ImgReshapeLayer(7, 7, false), pipeline);
      add(new ConvolutionLayer(1, 1, 25088, 4096)
          .setPaddingXY(0, 0)
          .setAndFree(hdf5.readDataSet("param_0", "layer_32")
              .permuteDimensionsAndFree(fullyconnectedOrder)), pipeline);
    }

    add(new ImgBandBiasLayer(4096)
        .setAndFree((hdf5.readDataSet("param_1", "layer_32"))), pipeline);
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  public void phase3(PipelineNetwork pipeline) {
    phase3a(pipeline);
    phase3b(pipeline);
  }

  public void phase3a(PipelineNetwork pipeline) {
    add(new ConvolutionLayer(1, 1, 4096, 4096)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_34")
            .permuteDimensionsAndFree(fullyconnectedOrder)), pipeline);
    add(new ImgBandBiasLayer(4096)
        .setAndFree((hdf5.readDataSet("param_1", "layer_34"))), pipeline);
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);

    add(new ConvolutionLayer(1, 1, 4096, 1000)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_36")
            .permuteDimensionsAndFree(fullyconnectedOrder)), pipeline);
    add(new ImgBandBiasLayer(1000)
        .setAndFree((hdf5.readDataSet("param_1", "layer_36"))), pipeline);
  }

  public void addPoolingLayer(final int size, PipelineNetwork pipeline) {
    if (large) {
      add(new ImgModulusPaddingLayer(size, size), pipeline);
    } else {
      add(new ImgModulusPaddingLayer(-size, -size), pipeline);
    }
    add(new PoolingLayer()
        .setMode(PoolingLayer.PoolingMode.Max)
        .setWindowXY(size, size)
        .setStrideXY(size, size), pipeline);
  }

  public void addConvolutionLayer(final int radius, final int inputBands, final int outputBands, final ActivationLayer.Mode activationMode, final String hdf_group, PipelineNetwork pipeline) {
    add(new ConvolutionLayer(radius, radius, inputBands, outputBands)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", hdf_group)
            .permuteDimensionsAndFree(convolutionOrder)), pipeline);
    add(new ImgBandBiasLayer(outputBands)
        .setAndFree((hdf5.readDataSet("param_1", hdf_group))), pipeline);
    add(new ActivationLayer(activationMode), pipeline);
  }

  public void phase3b(PipelineNetwork pipeline) {
    add(new SoftmaxActivationLayer()
        .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
        .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL), pipeline);
    add(new BandReducerLayer()
        .setMode(getFinalPoolingMode()), pipeline);
  }

  public Hdf5Archive getHDF5() {
    return hdf5;
  }

  public boolean isLarge() {
    return large;
  }

  public VGG16_HDF5 setLarge(boolean large) {
    this.large = large;
    return this;
  }

  public boolean isDense() {
    return dense;
  }

  public VGG16_HDF5 setDense(boolean dense) {
    this.dense = dense;
    return this;
  }

  public PoolingLayer.PoolingMode getFinalPoolingMode() {
    return finalPoolingMode;
  }

  public VGG16_HDF5 setFinalPoolingMode(PoolingLayer.PoolingMode finalPoolingMode) {
    this.finalPoolingMode = finalPoolingMode;
    return this;
  }

}
