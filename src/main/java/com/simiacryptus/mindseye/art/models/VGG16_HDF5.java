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

import com.simiacryptus.mindseye.art.util.Hdf5Archive;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.ref.wrappers.RefString;
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

  public VGG16_HDF5(Hdf5Archive hdf5) {
    this.hdf5 = hdf5;
  }

  @Nonnull
  public static VGG16_HDF5 fromHDF5() {
    try {
      return fromHDF5(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")));
    } catch (@Nonnull IOException | KeyManagementException | NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  public static VGG16_HDF5 fromHDF5(@Nonnull final File hdf) {
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
          logger.info(RefString.format("Exploded %s to %s (%s nodes)", layer.getName(),
              explode.getClass().getSimpleName(), ((DAGNetwork) explode).getNodes().size()));
        } else {
          logger.info(RefString.format("Exploded %s to %s (%s nodes)", layer.getName(),
              explode.getClass().getSimpleName(), explode.getName()));
        }
        return add(explode, model);
      } finally {
        layer.freeRef();
      }
    } else {
      model.add(layer).freeRef();
      return layer;
    }
  }

  public void phase0b(@Nonnull PipelineNetwork pipeline) {
    //add(new ImgMinSizeLayer(226, 226), pipeline);
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(new Tensor(-103.939, -116.779, -123.68));
    add(imgBandBiasLayer.addRef(), pipeline);
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1", pipeline);
  }

  public void phase1a(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3", pipeline);
  }

  public void phase1b2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 128, 128, ActivationLayer.Mode.RELU, "layer_8", pipeline);
  }

  public void phase1b1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 64, 128, ActivationLayer.Mode.RELU, "layer_6", pipeline);
  }

  public void phase1c3(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15", pipeline);
  }

  public void phase1c2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13", pipeline);
  }

  public void phase1c1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11", pipeline);
  }

  public void phase1d3(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22", pipeline);
  }

  public void phase1d2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_20", pipeline);
  }

  public void phase1d1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_18", pipeline);
  }

  public void phase1e3(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29", pipeline);
  }

  public void phase1e2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_27", pipeline);
  }

  public void phase1e1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_25", pipeline);
  }

  public void phase2(@Nonnull PipelineNetwork pipeline) {
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    addPoolingLayer(2, pipeline);
    add(new ImgModulusPaddingLayer(7, 7), pipeline);
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(7, 7, 512, 4096).setStrideXY(1, 1).setPaddingXY(0, 0);
    convolutionLayer.set(hdf5.readDataSet("param_0", "layer_32").reshapeCast(7, 7, 512, 4096).permuteDimensions(0, 1, 3, 2));
    add(convolutionLayer.addRef(), pipeline);
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(4096);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", "layer_32"));
    add(imgBandBiasLayer.addRef(), pipeline);
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  public void phase3a(@Nonnull PipelineNetwork pipeline) {
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 4096, 4096).setPaddingXY(0, 0);
    convolutionLayer.set(hdf5.readDataSet("param_0", "layer_34").permuteDimensions(fullyconnectedOrder));
    add(convolutionLayer.addRef(), pipeline);
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(4096);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", "layer_34"));
    add(imgBandBiasLayer.addRef(), pipeline);
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  public void addPoolingLayer(final int size, @Nonnull PipelineNetwork pipeline) {
    add(new ImgModulusPaddingLayer(size, size), pipeline);
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    PoolingLayer poolingLayer2 = poolingLayer.addRef();
    poolingLayer2.setWindowXY(size, size);
    PoolingLayer poolingLayer1 = poolingLayer2.addRef();
    poolingLayer1.setStrideXY(size, size);
    add(poolingLayer1.addRef(),
        pipeline);
  }

  public void addConvolutionLayer(final int radius, final int inputBands, final int outputBands,
                                  @Nonnull final ActivationLayer.Mode activationMode, final String hdf_group, @Nonnull PipelineNetwork pipeline) {
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPaddingXY(0, 0);
    convolutionLayer.set(hdf5.readDataSet("param_0", hdf_group).permuteDimensions(convolutionOrder));
    add(convolutionLayer.addRef(), pipeline);
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(outputBands);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", hdf_group));
    add(imgBandBiasLayer.addRef(), pipeline);
    add(new ActivationLayer(activationMode), pipeline);
  }

  public void phase3b(@Nonnull PipelineNetwork pipeline) {
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 4096, 1000).setPaddingXY(0, 0);
    convolutionLayer.set(hdf5.readDataSet("param_0", "layer_36").permuteDimensions(fullyconnectedOrder));
    add(convolutionLayer.addRef(), pipeline);
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(1000);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", "layer_36"));
    add(imgBandBiasLayer.addRef(), pipeline);
  }

  public void phase3c(@Nonnull PipelineNetwork pipeline) {
    SoftmaxActivationLayer softmaxActivationLayer = new SoftmaxActivationLayer();
    softmaxActivationLayer.setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE);
    SoftmaxActivationLayer softmaxActivationLayer1 = softmaxActivationLayer.addRef();
    softmaxActivationLayer1.setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL);
    add(softmaxActivationLayer1.addRef(), pipeline);
    BandReducerLayer bandReducerLayer = new BandReducerLayer();
    bandReducerLayer.setMode(PoolingLayer.PoolingMode.Max);
    add(bandReducerLayer.addRef(), pipeline);
  }

}
