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
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
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
  public static VGG19_HDF5 fromHDF5() {
    try {
      return fromHDF5(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg19_weights.h5")));
    } catch (@Nonnull IOException | KeyManagementException | NoSuchAlgorithmException e) {
      throw Util.throwException(e);
    }
  }

  @Nonnull
  public static VGG19_HDF5 fromHDF5(@Nonnull final File hdf) {
    try {
      return new VGG19_HDF5(new Hdf5Archive(hdf));
    } catch (Throwable e) {
      throw Util.throwException(e);
    }
  }

  @Nonnull
  protected static void add(@Nonnull Layer layer, @Nonnull PipelineNetwork model) {
    if (layer instanceof Explodable) {
      Layer explode = ((Explodable) layer).explode();
      try {
        if (explode instanceof DAGNetwork) {
          RefList<DAGNode> nodes = ((DAGNetwork) explode).getNodes();
          log.info(RefString.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(),
              nodes.size()));
          nodes.freeRef();
        } else {
          log.info(RefString.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(),
              explode.getName()));
        }
        add(explode, model);
      } finally {
        layer.freeRef();
      }
    } else {
      model.add(layer).freeRef();
      model.freeRef();
    }
  }

  public void phase0(@Nonnull PipelineNetwork pipeline) {
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(new Tensor(-103.939, -116.779, -123.68));
    add(imgBandBiasLayer, pipeline.addRef());
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1", pipeline);
  }

  public void phase1a(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3", pipeline);
  }

  public void phase1b2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 128, 128, ActivationLayer.Mode.RELU, "layer_8", pipeline);
  }

  public void phase1b1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline.addRef());
    addConvolutionLayer(3, 64, 128, ActivationLayer.Mode.RELU, "layer_6", pipeline);
  }

  public void phase1c4(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_17", pipeline);
  }

  public void phase1c3(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15", pipeline);
  }

  public void phase1c2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13", pipeline);
  }

  public void phase1c1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline.addRef());
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11", pipeline);
  }

  public void phase1d4(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_26", pipeline);
  }

  public void phase1d3(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_24", pipeline);
  }

  public void phase1d2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22", pipeline);
  }

  public void phase1d1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline.addRef());
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_20", pipeline);
  }

  public void phase1e4(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_35", pipeline);
  }

  public void phase1e3(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_33", pipeline);
  }

  public void phase1e2(@Nonnull PipelineNetwork pipeline) {
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_31", pipeline);
  }

  public void phase1e1(@Nonnull PipelineNetwork pipeline) {
    addPoolingLayer(2, pipeline.addRef());
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29", pipeline);
  }

  public void phase2(@Nonnull PipelineNetwork pipeline) {
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    addPoolingLayer(2, pipeline.addRef());
    add(new ImgModulusPaddingLayer(7, 7), pipeline.addRef());

    ConvolutionLayer convolutionLayer = new ConvolutionLayer(7, 7, 512, 4096);
    convolutionLayer.setStrideXY(1, 1);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(readShapeAndPermute("param_0", "layer_38", new int[]{7, 7, 512, 4096}, new int[]{0, 1, 3, 2}));
    add(convolutionLayer, pipeline.addRef());

    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(4096);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", "layer_38"));
    add(imgBandBiasLayer, pipeline.addRef());
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  @NotNull
  public Tensor readShapeAndPermute(String datasetName, String groups, int[] dims, int[] order) {
    Tensor tensor = hdf5.readDataSet(datasetName, groups);
    Tensor reshapeCast = tensor.reshapeCast(dims);
    tensor.freeRef();
    Tensor permuteDimensions = reshapeCast.permuteDimensions(order);
    reshapeCast.freeRef();
    return permuteDimensions;
  }

  public void phase3a(@Nonnull PipelineNetwork pipeline) {
    ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(1, 1, 4096, 4096);
    convolutionLayer1.setPaddingXY(0, 0);
    convolutionLayer1.set(readPermuted("param_0", fullyconnectedOrder, "layer_40"));
    add(convolutionLayer1, pipeline.addRef());
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(4096);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", "layer_40"));
    add(imgBandBiasLayer, pipeline.addRef());
    add(new ActivationLayer(ActivationLayer.Mode.RELU), pipeline);
  }

  public void addPoolingLayer(final int size, @Nonnull PipelineNetwork pipeline) {
    add(new ImgModulusPaddingLayer(size, size), pipeline.addRef());
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    poolingLayer.setWindowXY(size, size);
    poolingLayer.setStrideXY(size, size);
    add(poolingLayer, pipeline);
  }

  public void addConvolutionLayer(final int radius, final int inputBands, final int outputBands,
                                  @Nonnull final ActivationLayer.Mode activationMode, final String hdf_group, @Nonnull PipelineNetwork pipeline) {
    ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(radius, radius, inputBands, outputBands);
    convolutionLayer1.setPaddingXY(0, 0);
    convolutionLayer1.set(readPermuted("param_0", convolutionOrder, hdf_group));
    add(convolutionLayer1, pipeline.addRef());
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(outputBands);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", hdf_group));
    add(imgBandBiasLayer, pipeline.addRef());
    add(new ActivationLayer(activationMode), pipeline);
  }

  public void phase3b(@Nonnull PipelineNetwork pipeline) {
    ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(1, 1, 4096, 1000);
    convolutionLayer1.setPaddingXY(0, 0);
    convolutionLayer1.set(readPermuted("param_0", fullyconnectedOrder, "layer_42"));
    add(convolutionLayer1, pipeline.addRef());
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(1000);
    imgBandBiasLayer.set(hdf5.readDataSet("param_1", "layer_42"));
    add(imgBandBiasLayer, pipeline.addRef());
    SoftmaxActivationLayer softmaxActivationLayer = new SoftmaxActivationLayer();
    softmaxActivationLayer.setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE);
    softmaxActivationLayer.setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL);
    add(softmaxActivationLayer, pipeline.addRef());
    BandReducerLayer bandReducerLayer = new BandReducerLayer();
    bandReducerLayer.setMode(PoolingLayer.PoolingMode.Max);
    add(bandReducerLayer, pipeline);
  }

  @NotNull
  private Tensor readPermuted(String datasetName, int[] permutationOrder, String... groups) {
    Tensor dataSet = hdf5.readDataSet(datasetName, groups);
    Tensor permuteDimensions = dataSet.permuteDimensions(permutationOrder);
    dataSet.freeRef();
    return permuteDimensions;
  }

}
