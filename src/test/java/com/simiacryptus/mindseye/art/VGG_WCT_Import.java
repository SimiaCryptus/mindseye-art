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

import com.simiacryptus.mindseye.art.photo.FastPhotoStyleTransfer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.layers.LoggingWrapperLayer;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.PhotoUnpoolingLayer;
import com.simiacryptus.mindseye.layers.java.UnpoolingLayer;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/* To extract weights from FastPhotoStyle pretrained models, run this python script to export the torch data files as simple text:

``` python
import torchfile
import numpy

def convertNet(netId) :
    for i, module in enumerate(torchfile.load('models/vgg_normalised_%s_mask.t7' % netId).modules):
        if module.weight is not None:
            file = open('models/vgg_%s_' % netId + str(i) + '_weight.txt', 'w')
            file.write(numpy.array2string(module.weight, threshold=2147483646))
            file.close()
        if module.bias is not None:
            file = open('models/vgg_%s_' % netId + str(i) + '_bias.txt', 'w')
            file.write(numpy.array2string(module.bias, threshold=2147483646))
            file.close()

convertNet('conv1_1')
convertNet('conv2_1')
convertNet('conv3_1')
convertNet('conv4_1')
convertNet('conv5_1')

def convertInv(netId) :
    for i, module in enumerate(torchfile.load('models/feature_invertor_%s_mask.t7' % netId).modules):
        if module.weight is not None:
            file = open('models/inv_%s_' % netId + str(i) + '_weight.txt', 'w')
            file.write(numpy.array2string(module.weight, threshold=2147483646))
            file.close()
        if module.bias is not None:
            file = open('models/inv_%s_' % netId + str(i) + '_bias.txt', 'w')
            file.write(numpy.array2string(module.bias, threshold=2147483646))
            file.close()


convertInv('conv1_1')
convertInv('conv2_1')
convertInv('conv3_1')
convertInv('conv4_1')
convertInv('conv5_1')
```
 */
public class VGG_WCT_Import {

  public static final Logger log = LoggerFactory.getLogger(VGG_WCT_Import.class);
  private static final String fileBase = "H:\\SimiaCryptus\\data-science-tools\\FastPhotoStyle\\models\\";
  @Nonnull
  private static int[] convolutionOrder = {1, 0, 3, 2};
  private static boolean verbose = false;
  private static boolean simple = false;

  @Nonnull
  public static Layer encode_1() {
    PipelineNetwork pipeline = new PipelineNetwork(1);
    final String prefix1 = "vgg_conv1_1_0_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();
    final String prefix2 = "vgg_conv1_1_2_";
    pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 3, 64)).freeRef();
    return pipeline;
  }

  @Nonnull
  public static Layer encode_2() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "vgg_conv2_1_0_";
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
      convolutionLayer.setPaddingXY(0, 0);
      convolutionLayer.set(getWeight(prefix1));
      pipeline.add(convolutionLayer).freeRef();
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
      imgBandBiasLayer.set(getBias(prefix1));
      pipeline.add(imgBandBiasLayer).freeRef();
      final String prefix2 = "vgg_conv2_1_2_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 3, 64)).freeRef();
    }

    final String prefix1 = "vgg_conv2_1_5_";
    final String prefix2 = "vgg_conv2_1_9_";
    pipeline.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 64, 64)).freeRef();
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer).freeRef();
    pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 64, 128)).freeRef();

    return polish(pipeline);
  }

  @Nonnull
  public static Layer encode_3() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "vgg_conv3_1_0_";
      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(1, 1, 3, 3);
      convolutionLayer1.setPaddingXY(0, 0);
      convolutionLayer1.set(getWeight(prefix1));
      pipeline.add(convolutionLayer1).freeRef();
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
      imgBandBiasLayer.set(getBias(prefix1));
      pipeline.add(imgBandBiasLayer).freeRef();
      final String prefix2 = "vgg_conv3_1_2_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 3, 64)).freeRef();
    }

    {
      final String prefix1 = "vgg_conv3_1_5_";
      pipeline.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 64, 64)).freeRef();
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
      pipeline.add(poolingLayer).freeRef();
      final String prefix2 = "vgg_conv3_1_9_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 64, 128)).freeRef();
    }

    final String prefix1 = "vgg_conv3_1_12_";
    pipeline.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 128, 128)).freeRef();
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer).freeRef();
    final String prefix2 = "vgg_conv3_1_16_";
    pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 128, 256)).freeRef();

    return polish(pipeline);
  }

  @Nonnull
  public static Layer encode_4() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    final String prefix1 = "vgg_conv4_1_0_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();

    pipeline.add(convolutionLayer("vgg_conv4_1_2_", 3, 64)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_5_", 64, 64)).freeRef();
    PoolingLayer poolingLayer2 = new PoolingLayer();
    poolingLayer2.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer2).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_9_", 64, 128)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_12_", 128, 128)).freeRef();
    PoolingLayer poolingLayer1 = new PoolingLayer();
    poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer1).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_16_", 128, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_19_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_22_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_25_", 256, 256)).freeRef();
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer).freeRef();
    pipeline.add(convolutionLayer("vgg_conv4_1_29_", 256, 512)).freeRef();

    return polish(pipeline);
  }

  @Nonnull
  public static Layer encode_5() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    final String prefix1 = "vgg_conv5_1_0_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();

    pipeline.add(convolutionLayer("vgg_conv5_1_2_", 3, 64)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_5_", 64, 64)).freeRef();
    PoolingLayer poolingLayer3 = new PoolingLayer();
    poolingLayer3.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer3).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_9_", 64, 128)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_12_", 128, 128)).freeRef();
    PoolingLayer poolingLayer2 = new PoolingLayer();
    poolingLayer2.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer2).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_16_", 128, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_19_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_22_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_25_", 256, 256)).freeRef();
    PoolingLayer poolingLayer1 = new PoolingLayer();
    poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer1).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_29_", 256, 512)).freeRef();

    pipeline.add(convolutionLayer("vgg_conv5_1_32_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_35_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_38_", 512, 512)).freeRef();
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_42_", 512, 512)).freeRef();
    return polish(pipeline);
  }

  @Nonnull
  public static Layer decode_1() {
    PipelineNetwork pipeline = new PipelineNetwork(1);
    final String prefix1 = "inv_conv1_1_1_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();
    return pipeline;
  }

  @Nonnull
  public static Layer decode_2() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "inv_conv2_1_1_";
      pipeline.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 128, 64)).freeRef();
      pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
      final String prefix2 = "inv_conv2_1_5_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 64, 64)).freeRef();
    }

    final String prefix1 = "inv_conv2_1_8_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();

    return polish(pipeline);
  }

  @Nonnull
  public static Layer decode_3() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "inv_conv3_1_1_";
      pipeline.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 256, 128)).freeRef();
      pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
      final String prefix2 = "inv_conv3_1_5_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 128, 128)).freeRef();
    }

    {
      final String prefix1 = "inv_conv3_1_8_";
      pipeline.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 128, 64)).freeRef();
      pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
      final String prefix2 = "inv_conv3_1_12_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 64, 64)).freeRef();
    }

    final String prefix1 = "inv_conv3_1_15_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();

    return polish(pipeline);
  }

  @Nonnull
  public static Layer decode_4() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    pipeline.add(convolutionLayer("inv_conv4_1_1_", 512, 256)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_5_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_8_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_11_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_14_", 256, 128)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_18_", 128, 128)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_21_", 128, 64)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv4_1_25_", 64, 64)).freeRef();

    final String prefix1 = "inv_conv4_1_28_";
    ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer1.setPaddingXY(0, 0);
    convolutionLayer1.set(getWeight(prefix1));
    pipeline.add(convolutionLayer1).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();

    return polish(pipeline);
  }

  @Nonnull
  public static Layer decode_5() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    pipeline.add(convolutionLayer("inv_conv5_1_1_", 512, 512)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_5_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_8_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_11_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_14_", 512, 256)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_18_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_21_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_24_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_27_", 256, 128)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_31_", 128, 128)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_34_", 128, 64)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_38_", 64, 64)).freeRef();

    final String prefix1 = "inv_conv5_1_41_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();

    return polish(pipeline);
  }

  public static @Nonnull
  Layer photo_decode_2() {

    PipelineNetwork pipeline2 = new PipelineNetwork(2);

    {
      final String prefix1 = "vgg_conv2_1_0_";
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
      convolutionLayer.setPaddingXY(0, 0);
      convolutionLayer.set(getWeight(prefix1));
      pipeline2.add(convolutionLayer, pipeline2.getInput(1)).freeRef();
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
      imgBandBiasLayer.set(getBias(prefix1));
      pipeline2.add(imgBandBiasLayer).freeRef();
      final String prefix2 = "vgg_conv2_1_2_";
      pipeline2.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 3, 64)).freeRef();
    }

    final InnerNode prepool_1;
    {
      final String prefix1 = "vgg_conv2_1_5_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      prepool_1 = pipeline2.add(convolutionLayer(bias1, weight1, 64, 64));
    }

    {
      final String prefix1 = "inv_conv2_1_1_";
      final InnerNode wrap = pipeline2.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 128, 64), pipeline2.getInput(0));
      pipeline2.add(new PhotoUnpoolingLayer(), wrap, prepool_1).freeRef();
      final String prefix2 = "inv_conv2_1_5_";
      pipeline2.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 64, 64)).freeRef();
    }

    PipelineNetwork pipeline = new PipelineNetwork(1);
    final String prefix1 = "inv_conv2_1_8_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();
    pipeline2.add(pipeline).freeRef();
    return polish(pipeline2);
  }

  public static @Nonnull
  Layer photo_decode_3() {
    PipelineNetwork pipeline2 = new PipelineNetwork(2);

    {
      PipelineNetwork pipeline = new PipelineNetwork(1);
      final String prefix1 = "vgg_conv3_1_0_";
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
      convolutionLayer.setPaddingXY(0, 0);
      convolutionLayer.set(getWeight(prefix1));
      pipeline.add(convolutionLayer).freeRef();
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
      imgBandBiasLayer.set(getBias(prefix1));
      pipeline.add(imgBandBiasLayer).freeRef();
      final String prefix2 = "vgg_conv3_1_2_";
      pipeline.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 3, 64)).freeRef();
      pipeline2.add(pipeline, pipeline2.getInput(1)).freeRef();
    }

    final InnerNode prepool_1;
    {
      final String prefix1 = "vgg_conv3_1_5_";
      prepool_1 = pipeline2.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 64, 64));
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
      pipeline2.add(poolingLayer).freeRef();
      final String prefix2 = "vgg_conv3_1_9_";
      pipeline2.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 64, 128)).freeRef();
    }

    final InnerNode prepool_2;
    {
      final String prefix1 = "vgg_conv3_1_12_";
      prepool_2 = pipeline2.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 128, 128));
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
      pipeline2.add(poolingLayer).freeRef();
      final String prefix2 = "vgg_conv3_1_16_";
      pipeline2.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 128, 256)).freeRef();
    }

    {
      final String prefix1 = "inv_conv3_1_1_";
      final String prefix2 = "inv_conv3_1_5_";
      pipeline2.add(convolutionLayer(getBias(prefix1), getWeight(prefix1), 256, 128), pipeline2.getInput(0)).freeRef();
      pipeline2.add(new PhotoUnpoolingLayer(), pipeline2.getHead(), prepool_2).freeRef();
      pipeline2.add(convolutionLayer(getBias(prefix2), getWeight(prefix2), 128, 128)).freeRef();
    }

    {
      final String prefix1 = "inv_conv3_1_8_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv3_1_12_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      pipeline2.add(convolutionLayer(bias1, weight1, 128, 64)).freeRef();
      pipeline2.add(new PhotoUnpoolingLayer(), pipeline2.getHead(), prepool_1).freeRef();
      pipeline2.add(convolutionLayer(bias2, weight2, 64, 64)).freeRef();
    }

    PipelineNetwork pipeline = new PipelineNetwork(1);
    final String prefix1 = "inv_conv3_1_15_";
    final Tensor bias1 = getBias(prefix1);
    final Tensor weight1 = getWeight(prefix1);
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(weight1);
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(bias1);
    pipeline.add(imgBandBiasLayer).freeRef();
    pipeline2.add(pipeline).freeRef();

    return polish(pipeline2);
  }

  public static @Nonnull
  Layer photo_decode_4() {
    PipelineNetwork pipeline2 = new PipelineNetwork(2);

    {
      final String prefix1 = "vgg_conv4_1_0_";
      PipelineNetwork pipeline = new PipelineNetwork(1);
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, 3, 3);
      convolutionLayer.setPaddingXY(0, 0);
      convolutionLayer.set(getWeight(prefix1));
      pipeline.add(convolutionLayer).freeRef();
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
      imgBandBiasLayer.set(getBias(prefix1));
      pipeline.add(imgBandBiasLayer).freeRef();
      pipeline.add(convolutionLayer("vgg_conv4_1_2_", 3, 64)).freeRef();
      pipeline2.add(pipeline, pipeline2.getInput(1)).freeRef();
    }

    final InnerNode prepool_1 = pipeline2.add(convolutionLayer("vgg_conv4_1_5_", 64, 64));
    PoolingLayer poolingLayer1 = new PoolingLayer();
    poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
    pipeline2.add(poolingLayer1).freeRef();
    pipeline2.add(convolutionLayer("vgg_conv4_1_9_", 64, 128)).freeRef();
    final InnerNode prepool_2 = pipeline2.add(convolutionLayer("vgg_conv4_1_12_", 128, 128));
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    pipeline2.add(poolingLayer).freeRef();
    pipeline2.add(convolutionLayer("vgg_conv4_1_16_", 128, 256)).freeRef();
    pipeline2.add(convolutionLayer("vgg_conv4_1_19_", 256, 256)).freeRef();
    pipeline2.add(convolutionLayer("vgg_conv4_1_22_", 256, 256)).freeRef();
    final InnerNode prepool_3 = pipeline2.add(convolutionLayer("vgg_conv4_1_25_", 256, 256));
    pipeline2.add(convolutionLayer("inv_conv4_1_1_", 512, 256), pipeline2.getInput(0)).freeRef();
    pipeline2.add(new PhotoUnpoolingLayer(), pipeline2.getHead(), prepool_3).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_5_", 256, 256)).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_8_", 256, 256)).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_11_", 256, 256)).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_14_", 256, 128)).freeRef();
    pipeline2.add(new PhotoUnpoolingLayer(), pipeline2.getHead(), prepool_2).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_18_", 128, 128)).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_21_", 128, 64)).freeRef();
    pipeline2.add(new PhotoUnpoolingLayer(), pipeline2.getHead(), prepool_1).freeRef();
    pipeline2.add(convolutionLayer("inv_conv4_1_25_", 64, 64)).freeRef();

    final String prefix1 = "inv_conv4_1_28_";
    PipelineNetwork pipeline = new PipelineNetwork(1);
    ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer1.setPaddingXY(0, 0);
    convolutionLayer1.set(getWeight(prefix1));
    pipeline.add(convolutionLayer1).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();
    pipeline2.add(pipeline).freeRef();

    return polish(pipeline2);
  }

  public static @Nonnull
  Layer photo_decode_5() {
    PipelineNetwork pipeline = new PipelineNetwork(2);

    {
      final String prefix1 = "vgg_conv5_1_0_";
      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(1, 1, 3, 3);
      convolutionLayer1.setPaddingXY(0, 0);
      convolutionLayer1.set(getWeight(prefix1));
      pipeline.add(convolutionLayer1, pipeline.getInput(1)).freeRef();
      ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
      imgBandBiasLayer.set(getBias(prefix1));
      pipeline.add(imgBandBiasLayer).freeRef();
    }

    pipeline.add(convolutionLayer("vgg_conv5_1_2_", 3, 64)).freeRef();
    final InnerNode prepool_1 = pipeline.add(convolutionLayer("vgg_conv5_1_5_", 64, 64));
    PoolingLayer poolingLayer2 = new PoolingLayer();
    poolingLayer2.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer2).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_9_", 64, 128)).freeRef();
    final InnerNode prepool_2 = pipeline.add(convolutionLayer("vgg_conv5_1_12_", 128, 128));
    PoolingLayer poolingLayer1 = new PoolingLayer();
    poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer1).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_16_", 128, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_19_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_22_", 256, 256)).freeRef();
    final InnerNode prepool_3 = pipeline.add(convolutionLayer("vgg_conv5_1_25_", 256, 256));
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
    pipeline.add(poolingLayer).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_29_", 256, 512)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_32_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("vgg_conv5_1_35_", 512, 512)).freeRef();
    final InnerNode prepool_4 = pipeline.add(convolutionLayer("vgg_conv5_1_38_", 512, 512));

    pipeline.add(convolutionLayer("inv_conv5_1_1_", 512, 512), pipeline.getInput(0)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_4).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_5_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_8_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_11_", 512, 512)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_14_", 512, 256)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_3).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_18_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_21_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_24_", 256, 256)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_27_", 256, 128)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_2).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_31_", 128, 128)).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_34_", 128, 64)).freeRef();
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_1).freeRef();
    pipeline.add(convolutionLayer("inv_conv5_1_38_", 64, 64)).freeRef();

    final String prefix1 = "inv_conv5_1_41_";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 64, 3);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(getWeight(prefix1));
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(3);
    imgBandBiasLayer.set(getBias(prefix1));
    pipeline.add(imgBandBiasLayer).freeRef();
    return polish(pipeline);
  }

  @Nonnull
  public static Tensor getBias(String prefix1) {
    return loadNumpyTensor(fileBase + prefix1 + "bias.txt");
  }

  @Nonnull
  public static Tensor getWeight(String prefix4) {
    Tensor tensor = loadNumpyTensor(fileBase + prefix4 + "weight.txt");
    Tensor permuteDimensions = tensor.permuteDimensions(convolutionOrder);
    tensor.freeRef();
    return permuteDimensions;
  }

  @Nonnull
  public static Tensor loadNumpyTensor(@Nonnull String file) {
    try {
      Object parse = parse(FileUtils.readFileToString(new File(file), "UTF-8"));
      return new Tensor(toStream(parse).toArray(), Tensor.reverse(dims(parse)));
    } catch (IOException e) {
      throw Util.throwException(e);
    }
  }

  @Nonnull
  public static FastPhotoStyleTransfer newFastPhotoStyleTransfer() {
    return new FastPhotoStyleTransfer(decode_1(), encode_1(), photo_decode_2(), encode_2(), photo_decode_3(),
        encode_3(), photo_decode_4(), encode_4());
  }

  @Nonnull
  private static PipelineNetwork convolutionLayer(String prefix4, int inBands, int outBands) {
    return convolutionLayer(getBias(prefix4), getWeight(prefix4), inBands, outBands);
  }

  @Nonnull
  private static Layer polish(@Nonnull Layer pipeline) {
    ((PipelineNetwork) pipeline).visitNodes(true, node -> {
      Layer layer = node.getLayer();
      assert layer != null;
      final String name = layer.getName();
      if (!simple && layer instanceof Explodable) {
        Layer explode = ((Explodable) layer).explode();
        layer.freeRef();
        layer = explode;
      }
      if (verbose) {
        Layer layer1 = new LoggingWrapperLayer(layer);
        layer1.setName(name);
        layer = layer1;
      }
      node.setLayer(layer);
      node.freeRef();
    });
    pipeline.freeze();
    if (verbose)
      pipeline = new LoggingWrapperLayer(pipeline);
    return pipeline;
  }

  @Nonnull
  private static PipelineNetwork convolutionLayer(@Nonnull PipelineNetwork pipeline, Tensor bias1, @Nonnull Tensor weight1, int inBands,
                                                  int outBands) {
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, inBands, outBands);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(weight1);
    pipeline.add(convolutionLayer).freeRef();
    ImgBandBiasLayer imgBandBiasLayer = new ImgBandBiasLayer(outBands);
    imgBandBiasLayer.set(bias1);
    pipeline.add(imgBandBiasLayer).freeRef();
    pipeline.add(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
    return pipeline;
  }

  @Nonnull
  private static PipelineNetwork convolutionLayer(Tensor bias1, @Nonnull Tensor weight1, int inBands, int outBands) {
    PipelineNetwork layer = new PipelineNetwork(1);
    layer.setName(RefString.format("Conv(%s/%s)", inBands, outBands));
    return convolutionLayer(
        layer, bias1, weight1, inBands, outBands);
  }

  @Nonnull
  private static RefDoubleStream toStream(Object data) {
    if (data instanceof Double) {
      return RefDoubleStream.of((Double) data);
    } else {
      return RefArrays.stream((Object[]) data).flatMapToDouble(x -> toStream(x));
    }
  }

  @RefIgnore
  private static int[] dims(@RefAware Object data) {
    try {
      if (data instanceof Double) {
        return new int[]{};
      } else {
        int length = Array.getLength(data);
        List<int[]> childDims = new ArrayList<>();
        for (int i = 0; i < length; i++) {
          childDims.add(dims(Array.get(data, i)));
        }
        int[] head = childDims.get(0);
        childDims.stream().forEach(d -> Arrays.equals(head, d));
        return IntStream.concat(IntStream.of(length), Arrays.stream(head)).toArray();
      }
    } finally {
      RefUtil.freeRef(data);
    }
  }

  private static Object parse(String data) {
    data = data.trim();
    if (data.startsWith("[")) {
      if (!data.endsWith("]"))
        throw new AssertionError();
      final String strippedString = data.substring(1, data.length() - 1);
      RefArrayList<String> splitBuffer = new RefArrayList<>();
      int parenBalence = 0;
      int lastCut = 0;
      for (int i = 0; i < strippedString.length(); i++) {
        final char c = strippedString.charAt(i);
        if (c == '[') {
          parenBalence++;
        } else if (c == ']') {
          if (0 == --parenBalence) {
            splitBuffer.add(strippedString.substring(lastCut, i + 1));
            lastCut = i + 1;
          }
        } else if (0 == parenBalence && (c == ',' || c == ' ')) {
          splitBuffer.add(strippedString.substring(lastCut, i + 1));
          lastCut = i + 1;
        }
      }
      splitBuffer.add(strippedString.substring(lastCut, strippedString.length()));
      Object[] array = splitBuffer.stream().map(s -> s.trim()).filter(x -> !x.isEmpty()).map(data1 -> parse(data1)).toArray();
      splitBuffer.freeRef();
      return array;
    } else {
      return Double.parseDouble(data);
    }
  }
}
