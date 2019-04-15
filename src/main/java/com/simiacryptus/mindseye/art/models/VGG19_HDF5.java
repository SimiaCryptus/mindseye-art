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

class VGG19_HDF5 {

  public static final Logger log = LoggerFactory.getLogger(VGG19_HDF5.class);
  public final Hdf5Archive hdf5;
  @Nonnull
  int[] convolutionOrder = {3, 2, 0, 1};
  @Nonnull
  int[] fullyconnectedOrder = {1, 0};

  public VGG19_HDF5(Hdf5Archive hdf5) {
    this.hdf5 = hdf5;
  }

  @Nonnull
  protected static Layer add(@Nonnull Layer layer, @Nonnull PipelineNetwork model) {
    if (layer instanceof Explodable) {
      Layer explode = ((Explodable) layer).explode();
      try {
        if (explode instanceof DAGNetwork) {
          log.info(String.format(
              "Exploded %s to %s (%s nodes)",
              layer.getName(),
              explode.getClass().getSimpleName(),
              ((DAGNetwork) explode).getNodes().size()
          ));
        } else {
          log.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), explode.getName()));
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

  public static VGG19_HDF5 fromHDF5() {
    try {
      return fromHDF5(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg19_weights.h5")));
    } catch (IOException | KeyManagementException | NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }

  public static VGG19_HDF5 fromHDF5(final File hdf) {
    try {
      return new VGG19_HDF5(new Hdf5Archive(hdf));
    } catch (@Nonnull final RuntimeException e) {
      throw e;
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }

  public void phase0(PipelineNetwork pipeline) {
    //add(new ImgMinSizeLayer(226, 226), pipeline);
    add(new ImgBandBiasLayer(3).setAndFree(new Tensor(-103.939, -116.779, -123.68)), pipeline);
  }

  public void phase1a2(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3", pipeline);
  }

  public void phase1a1(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1", pipeline);
  }

  public void phase1b2(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 128, 128, ActivationLayer.Mode.RELU, "layer_8", pipeline);
  }

  public void phase1b1(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 64, 128, ActivationLayer.Mode.RELU, "layer_6", pipeline);
  }

  public void phase1c4(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_17", pipeline);
  }

  public void phase1c3(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15", pipeline);
  }

  public void phase1c2(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13", pipeline);
  }

  public void phase1c1(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11", pipeline);
  }

  public void phase1d4(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_26", pipeline);
  }

  public void phase1d3(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_24", pipeline);
  }

  public void phase1d2(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22", pipeline);
  }

  public void phase1d1(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_20", pipeline);
  }

  public void phase1e4(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_35", pipeline);
  }

  public void phase1e3(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_33", pipeline);
  }

  public void phase1e2(PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_31", pipeline);
  }

  public void phase1e1(PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29", pipeline);
  }

  public void phase2(PipelineNetwork pipeline) {
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    addPoolingLayer(2, pipeline);
    if (true) {
      add(new ImgModulusPaddingLayer(7, 7), pipeline);
    } else {
      add(new ImgModulusPaddingLayer(-7, -7), pipeline);
    }

    add(new ConvolutionLayer(7, 7, 512, 4096)
        .setStrideXY(1, 1)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_38")
            .reshapeCastAndFree(7, 7, 512, 4096).permuteDimensionsAndFree(0, 1, 3, 2)
        ), pipeline);

    add(new ImgBandBiasLayer(4096)
        .setAndFree((hdf5.readDataSet("param_1", "layer_38"))), pipeline);
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  public void phase3a(PipelineNetwork pipeline) {
    add(new ConvolutionLayer(1, 1, 4096, 4096)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_40")
            .permuteDimensionsAndFree(fullyconnectedOrder)), pipeline);
    add(new ImgBandBiasLayer(4096)
        .setAndFree((hdf5.readDataSet("param_1", "layer_40"))), pipeline);
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  public void addPoolingLayer(final int size, PipelineNetwork pipeline) {
    if (true) {
      add(new ImgModulusPaddingLayer(size, size), pipeline);
    } else {
      Layer layer = new ImgModulusPaddingLayer(-size, -size);
      add(layer, pipeline);
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
    add(new ConvolutionLayer(1, 1, 4096, 1000)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_42")
            .permuteDimensionsAndFree(fullyconnectedOrder)), pipeline);
    add(new ImgBandBiasLayer(1000)
        .setAndFree((hdf5.readDataSet("param_1", "layer_42"))), pipeline);
    add(new SoftmaxActivationLayer()
        .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
        .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL), pipeline);
    add(new BandReducerLayer()
        .setMode(PoolingLayer.PoolingMode.Max), pipeline);
  }


}
