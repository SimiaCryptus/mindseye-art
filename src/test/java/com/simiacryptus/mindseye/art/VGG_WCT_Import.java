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
import com.simiacryptus.ref.wrappers.*;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;

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
public @RefAware
class VGG_WCT_Import {

  public static final Logger log = LoggerFactory.getLogger(VGG_WCT_Import.class);
  private static final String fileBase = "H:\\SimiaCryptus\\data-science-tools\\FastPhotoStyle\\models\\";
  @Nonnull
  private static int[] convolutionOrder = {1, 0, 3, 2};
  private static boolean verbose = false;
  private static boolean simple = false;

  public static Layer encode_1() {
    PipelineNetwork pipeline = new PipelineNetwork(1);
    final String prefix1 = "vgg_conv1_1_0_";
    final Tensor weight1 = getWeight(prefix1);
    final Tensor bias1 = getBias(prefix1);
    final String prefix2 = "vgg_conv1_1_2_";
    final Tensor weight2 = getWeight(prefix2);
    final Tensor bias2 = getBias(prefix2);
    pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
    pipeline.add(new ImgBandBiasLayer(3).set(bias1)).freeRef();
    pipeline.add(convolutionLayer(bias2, weight2, 3, 64));
    return pipeline;
  }

  public static Layer encode_2() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "vgg_conv2_1_0_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv2_1_2_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(bias1)).freeRef();
      pipeline.add(convolutionLayer(bias2, weight2, 3, 64));
    }

    {
      final String prefix1 = "vgg_conv2_1_5_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv2_1_9_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      pipeline.add(convolutionLayer(bias1, weight1, 64, 64));
      pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline.add(convolutionLayer(bias2, weight2, 64, 128));
    }

    return polish(pipeline);

  }

  public static Layer encode_3() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "vgg_conv3_1_0_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv3_1_2_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(bias1)).freeRef();
      pipeline.add(convolutionLayer(bias2, weight2, 3, 64));
    }

    {
      final String prefix1 = "vgg_conv3_1_5_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv3_1_9_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      pipeline.add(convolutionLayer(bias1, weight1, 64, 64));
      pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline.add(convolutionLayer(bias2, weight2, 64, 128));
    }

    {
      final String prefix1 = "vgg_conv3_1_12_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv3_1_16_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);

      pipeline.add(convolutionLayer(bias1, weight1, 128, 128));
      pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline.add(convolutionLayer(bias2, weight2, 128, 256));
    }

    return polish(pipeline);

  }

  public static Layer encode_4() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "vgg_conv4_1_0_";
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(getBias(prefix1))).freeRef();
    }

    pipeline.add(convolutionLayer("vgg_conv4_1_2_", 3, 64));
    pipeline.add(convolutionLayer("vgg_conv4_1_5_", 64, 64));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv4_1_9_", 64, 128));
    pipeline.add(convolutionLayer("vgg_conv4_1_12_", 128, 128));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv4_1_16_", 128, 256));
    pipeline.add(convolutionLayer("vgg_conv4_1_19_", 256, 256));
    pipeline.add(convolutionLayer("vgg_conv4_1_22_", 256, 256));
    pipeline.add(convolutionLayer("vgg_conv4_1_25_", 256, 256));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv4_1_29_", 256, 512));

    return polish(pipeline);

  }

  public static Layer encode_5() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "vgg_conv5_1_0_";
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(getBias(prefix1))).freeRef();
    }

    pipeline.add(convolutionLayer("vgg_conv5_1_2_", 3, 64));
    pipeline.add(convolutionLayer("vgg_conv5_1_5_", 64, 64));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_9_", 64, 128));
    pipeline.add(convolutionLayer("vgg_conv5_1_12_", 128, 128));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_16_", 128, 256));
    pipeline.add(convolutionLayer("vgg_conv5_1_19_", 256, 256));
    pipeline.add(convolutionLayer("vgg_conv5_1_22_", 256, 256));
    pipeline.add(convolutionLayer("vgg_conv5_1_25_", 256, 256));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_29_", 256, 512));

    pipeline.add(convolutionLayer("vgg_conv5_1_32_", 512, 512));
    pipeline.add(convolutionLayer("vgg_conv5_1_35_", 512, 512));
    pipeline.add(convolutionLayer("vgg_conv5_1_38_", 512, 512));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_42_", 512, 512));

    return polish(pipeline);

  }

  public static Layer decode_1() {
    PipelineNetwork pipeline = new PipelineNetwork(1);
    final String prefix1 = "inv_conv1_1_1_";
    final Tensor bias1 = getBias(prefix1);
    final Tensor weight1 = getWeight(prefix1);
    pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
    Layer layer = new ImgBandBiasLayer(3).set(bias1);
    pipeline.add(layer).freeRef();
    return pipeline;
  }

  public static Layer decode_2() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "inv_conv2_1_1_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv2_1_5_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      pipeline.add(convolutionLayer(bias1, weight1, 128, 64));
      pipeline.add(new UnpoolingLayer(2, 2));
      pipeline.add(convolutionLayer(bias2, weight2, 64, 64));
    }

    {
      final String prefix1 = "inv_conv2_1_8_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(bias1);
      pipeline.add(layer).freeRef();
    }

    return polish(pipeline);
  }

  public static Layer decode_3() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    {
      final String prefix1 = "inv_conv3_1_1_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv3_1_5_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      pipeline.add(convolutionLayer(bias1, weight1, 256, 128));
      pipeline.add(new UnpoolingLayer(2, 2));
      pipeline.add(convolutionLayer(bias2, weight2, 128, 128));
    }

    {
      final String prefix1 = "inv_conv3_1_8_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv3_1_12_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      pipeline.add(convolutionLayer(bias1, weight1, 128, 64));
      pipeline.add(new UnpoolingLayer(2, 2));
      pipeline.add(convolutionLayer(bias2, weight2, 64, 64));
    }

    {
      final String prefix1 = "inv_conv3_1_15_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(bias1);
      pipeline.add(layer).freeRef();
    }

    return polish(pipeline);
  }

  public static Layer decode_4() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    pipeline.add(convolutionLayer("inv_conv4_1_1_", 512, 256));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv4_1_5_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv4_1_8_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv4_1_11_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv4_1_14_", 256, 128));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv4_1_18_", 128, 128));
    pipeline.add(convolutionLayer("inv_conv4_1_21_", 128, 64));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv4_1_25_", 64, 64));

    {
      final String prefix1 = "inv_conv4_1_28_";
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(getBias(prefix1));
      pipeline.add(layer).freeRef();
    }

    return polish(pipeline);
  }

  public static Layer decode_5() {
    PipelineNetwork pipeline = new PipelineNetwork(1);

    pipeline.add(convolutionLayer("inv_conv5_1_1_", 512, 512));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv5_1_5_", 512, 512));
    pipeline.add(convolutionLayer("inv_conv5_1_8_", 512, 512));
    pipeline.add(convolutionLayer("inv_conv5_1_11_", 512, 512));
    pipeline.add(convolutionLayer("inv_conv5_1_14_", 512, 256));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv5_1_18_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv5_1_21_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv5_1_24_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv5_1_27_", 256, 128));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv5_1_31_", 128, 128));
    pipeline.add(convolutionLayer("inv_conv5_1_34_", 128, 64));
    pipeline.add(new UnpoolingLayer(2, 2));
    pipeline.add(convolutionLayer("inv_conv5_1_38_", 64, 64));

    {
      final String prefix1 = "inv_conv5_1_41_";
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(getBias(prefix1));
      pipeline.add(layer).freeRef();
    }

    return polish(pipeline);
  }

  public static @NotNull Layer photo_decode_2() {

    PipelineNetwork pipeline2 = new PipelineNetwork(2);

    {
      final String prefix1 = "vgg_conv2_1_0_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv2_1_2_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      pipeline2.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(weight1), pipeline2.getInput(1)).freeRef();
      pipeline2.add(new ImgBandBiasLayer(3).set(bias1)).freeRef();
      pipeline2.add(convolutionLayer(bias2, weight2, 3, 64));
    }

    InnerNode prepool_1;
    {
      final String prefix1 = "vgg_conv2_1_5_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      prepool_1 = pipeline2.add(convolutionLayer(bias1, weight1, 64, 64));
    }

    {
      final String prefix1 = "inv_conv2_1_1_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv2_1_5_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      final InnerNode wrap = pipeline2.add(convolutionLayer(bias1, weight1, 128, 64), pipeline2.getInput(0));
      pipeline2.add(new PhotoUnpoolingLayer(2, 2), wrap, prepool_1).freeRef();
      pipeline2.add(convolutionLayer(bias2, weight2, 64, 64));
    }

    {
      PipelineNetwork pipeline = new PipelineNetwork(1);
      final String prefix1 = "inv_conv2_1_8_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(bias1);
      pipeline.add(layer).freeRef();
      pipeline2.add(pipeline);
    }

    return polish(pipeline2);
  }

  public static @NotNull Layer photo_decode_3() {
    PipelineNetwork pipeline2 = new PipelineNetwork(2);

    {
      PipelineNetwork pipeline = new PipelineNetwork(1);
      final String prefix1 = "vgg_conv3_1_0_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv3_1_2_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(bias1)).freeRef();
      pipeline.add(convolutionLayer(bias2, weight2, 3, 64));
      pipeline2.add(pipeline, pipeline2.getInput(1));
    }

    final InnerNode prepool_1;
    {
      final String prefix1 = "vgg_conv3_1_5_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv3_1_9_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);
      prepool_1 = pipeline2.add(convolutionLayer(bias1, weight1, 64, 64));
      pipeline2.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline2.add(convolutionLayer(bias2, weight2, 64, 128));
    }

    final InnerNode prepool_2;
    {
      final String prefix1 = "vgg_conv3_1_12_";
      final Tensor weight1 = getWeight(prefix1);
      final Tensor bias1 = getBias(prefix1);
      final String prefix2 = "vgg_conv3_1_16_";
      final Tensor weight2 = getWeight(prefix2);
      final Tensor bias2 = getBias(prefix2);

      prepool_2 = pipeline2.add(convolutionLayer(bias1, weight1, 128, 128));
      pipeline2.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      pipeline2.add(convolutionLayer(bias2, weight2, 128, 256));
    }

    {
      final String prefix1 = "inv_conv3_1_1_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv3_1_5_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      pipeline2.add(convolutionLayer(bias1, weight1, 256, 128), pipeline2.getInput(0));
      pipeline2.add(new PhotoUnpoolingLayer(2, 2), pipeline2.getHead(), prepool_2);
      pipeline2.add(convolutionLayer(bias2, weight2, 128, 128));
    }

    {
      final String prefix1 = "inv_conv3_1_8_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      final String prefix2 = "inv_conv3_1_12_";
      final Tensor bias2 = getBias(prefix2);
      final Tensor weight2 = getWeight(prefix2);
      pipeline2.add(convolutionLayer(bias1, weight1, 128, 64));
      pipeline2.add(new PhotoUnpoolingLayer(2, 2), pipeline2.getHead(), prepool_1);
      pipeline2.add(convolutionLayer(bias2, weight2, 64, 64));
    }

    {
      PipelineNetwork pipeline = new PipelineNetwork(1);
      final String prefix1 = "inv_conv3_1_15_";
      final Tensor bias1 = getBias(prefix1);
      final Tensor weight1 = getWeight(prefix1);
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(weight1)).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(bias1);
      pipeline.add(layer).freeRef();
      pipeline2.add(pipeline);
    }

    return polish(pipeline2);
  }

  public static @NotNull Layer photo_decode_4() {
    PipelineNetwork pipeline2 = new PipelineNetwork(2);

    {
      final String prefix1 = "vgg_conv4_1_0_";
      PipelineNetwork pipeline = new PipelineNetwork(1);
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(getBias(prefix1))).freeRef();
      pipeline.add(convolutionLayer("vgg_conv4_1_2_", 3, 64));
      pipeline2.add(pipeline, pipeline2.getInput(1));
    }

    final InnerNode prepool_1 = pipeline2.add(convolutionLayer("vgg_conv4_1_5_", 64, 64));
    pipeline2.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline2.add(convolutionLayer("vgg_conv4_1_9_", 64, 128));
    final InnerNode prepool_2 = pipeline2.add(convolutionLayer("vgg_conv4_1_12_", 128, 128));
    pipeline2.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline2.add(convolutionLayer("vgg_conv4_1_16_", 128, 256));
    pipeline2.add(convolutionLayer("vgg_conv4_1_19_", 256, 256));
    pipeline2.add(convolutionLayer("vgg_conv4_1_22_", 256, 256));
    final InnerNode prepool_3 = pipeline2.add(convolutionLayer("vgg_conv4_1_25_", 256, 256));
    pipeline2.add(convolutionLayer("inv_conv4_1_1_", 512, 256), pipeline2.getInput(0));
    pipeline2.add(new PhotoUnpoolingLayer(2, 2), pipeline2.getHead(), prepool_3);
    pipeline2.add(convolutionLayer("inv_conv4_1_5_", 256, 256));
    pipeline2.add(convolutionLayer("inv_conv4_1_8_", 256, 256));
    pipeline2.add(convolutionLayer("inv_conv4_1_11_", 256, 256));
    pipeline2.add(convolutionLayer("inv_conv4_1_14_", 256, 128));
    pipeline2.add(new PhotoUnpoolingLayer(2, 2), pipeline2.getHead(), prepool_2);
    pipeline2.add(convolutionLayer("inv_conv4_1_18_", 128, 128));
    pipeline2.add(convolutionLayer("inv_conv4_1_21_", 128, 64));
    pipeline2.add(new PhotoUnpoolingLayer(2, 2), pipeline2.getHead(), prepool_1);
    pipeline2.add(convolutionLayer("inv_conv4_1_25_", 64, 64));

    {
      final String prefix1 = "inv_conv4_1_28_";
      PipelineNetwork pipeline = new PipelineNetwork(1);
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(getBias(prefix1));
      pipeline.add(layer).freeRef();
      pipeline2.add(pipeline);
    }

    return polish(pipeline2);
  }

  public static @NotNull Layer photo_decode_5() {
    PipelineNetwork pipeline = new PipelineNetwork(2);

    {
      final String prefix1 = "vgg_conv5_1_0_";
      pipeline.add(new ConvolutionLayer(1, 1, 3, 3).setPaddingXY(0, 0).set(getWeight(prefix1)), pipeline.getInput(1))
          .freeRef();
      pipeline.add(new ImgBandBiasLayer(3).set(getBias(prefix1))).freeRef();
    }

    pipeline.add(convolutionLayer("vgg_conv5_1_2_", 3, 64));
    final InnerNode prepool_1 = pipeline.add(convolutionLayer("vgg_conv5_1_5_", 64, 64));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_9_", 64, 128));
    final InnerNode prepool_2 = pipeline.add(convolutionLayer("vgg_conv5_1_12_", 128, 128));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_16_", 128, 256));
    pipeline.add(convolutionLayer("vgg_conv5_1_19_", 256, 256));
    pipeline.add(convolutionLayer("vgg_conv5_1_22_", 256, 256));
    final InnerNode prepool_3 = pipeline.add(convolutionLayer("vgg_conv5_1_25_", 256, 256));
    pipeline.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    pipeline.add(convolutionLayer("vgg_conv5_1_29_", 256, 512));
    pipeline.add(convolutionLayer("vgg_conv5_1_32_", 512, 512));
    pipeline.add(convolutionLayer("vgg_conv5_1_35_", 512, 512));
    final InnerNode prepool_4 = pipeline.add(convolutionLayer("vgg_conv5_1_38_", 512, 512));

    pipeline.add(convolutionLayer("inv_conv5_1_1_", 512, 512), pipeline.getInput(0));
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_4);
    pipeline.add(convolutionLayer("inv_conv5_1_5_", 512, 512));
    pipeline.add(convolutionLayer("inv_conv5_1_8_", 512, 512));
    pipeline.add(convolutionLayer("inv_conv5_1_11_", 512, 512));
    pipeline.add(convolutionLayer("inv_conv5_1_14_", 512, 256));
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_3);
    pipeline.add(convolutionLayer("inv_conv5_1_18_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv5_1_21_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv5_1_24_", 256, 256));
    pipeline.add(convolutionLayer("inv_conv5_1_27_", 256, 128));
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_2);
    pipeline.add(convolutionLayer("inv_conv5_1_31_", 128, 128));
    pipeline.add(convolutionLayer("inv_conv5_1_34_", 128, 64));
    pipeline.add(new UnpoolingLayer(2, 2), pipeline.getHead(), prepool_1);
    pipeline.add(convolutionLayer("inv_conv5_1_38_", 64, 64));

    {
      final String prefix1 = "inv_conv5_1_41_";
      pipeline.add(new ConvolutionLayer(3, 3, 64, 3).setPaddingXY(0, 0).set(getWeight(prefix1))).freeRef();
      Layer layer = new ImgBandBiasLayer(3).set(getBias(prefix1));
      pipeline.add(layer).freeRef();
    }

    return polish(pipeline);
  }

  @NotNull
  public static Tensor getBias(String prefix1) {
    return loadNumpyTensor(fileBase + prefix1 + "bias.txt");
  }

  public static Tensor getWeight(String prefix4) {
    return loadNumpyTensor(fileBase + prefix4 + "weight.txt").permuteDimensions(convolutionOrder);
  }

  public static Tensor loadNumpyTensor(String file) {
    try {
      Object parse = parse(FileUtils.readFileToString(new File(file), "UTF-8"));
      return new Tensor(toStream(parse).toArray(), Tensor.reverse(dims(parse)));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static FastPhotoStyleTransfer newFastPhotoStyleTransfer() {
    return new FastPhotoStyleTransfer(decode_1(), encode_1(), photo_decode_2(), encode_2(), photo_decode_3(),
        encode_3(), photo_decode_4(), encode_4());
  }

  private static PipelineNetwork convolutionLayer(String prefix4, int inBands, int outBands) {
    return convolutionLayer(getBias(prefix4), getWeight(prefix4), inBands, outBands);
  }

  @NotNull
  private static Layer polish(Layer pipeline) {
    ((PipelineNetwork) pipeline).visitNodes(true, node -> {
      Layer layer = node.getLayer();
      final String name = layer.getName();
      if (!simple && layer instanceof Explodable) {
        layer = ((Explodable) layer).explode();
      }
      if (verbose)
        layer = new LoggingWrapperLayer(layer).setName(name);
      node.setLayer(layer);
    });
    pipeline.freeze();
    if (verbose)
      pipeline = new LoggingWrapperLayer(pipeline);
    return pipeline;
  }

  private static PipelineNetwork convolutionLayer(PipelineNetwork pipeline, Tensor bias1, Tensor weight1, int inBands,
                                                  int outBands) {
    pipeline.add(new ConvolutionLayer(3, 3, inBands, outBands).setPaddingXY(0, 0).set(weight1)).freeRef();
    pipeline.add(new ImgBandBiasLayer(outBands).set(bias1)).freeRef();
    pipeline.add(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
    return pipeline;
  }

  private static PipelineNetwork convolutionLayer(Tensor bias1, Tensor weight1, int inBands, int outBands) {
    return convolutionLayer(
        (PipelineNetwork) new PipelineNetwork(1).setName(String.format("Conv(%s/%s)", inBands, outBands)), bias1,
        weight1, inBands, outBands);
  }

  private static RefDoubleStream toStream(Object data) {
    if (data instanceof Double) {
      return RefDoubleStream.of((Double) data);
    } else {
      return RefArrays.stream(((Object[]) data)).flatMapToDouble(x -> toStream(x));
    }
  }

  private static int[] dims(Object data) {
    if (data instanceof Double) {
      return new int[]{};
    } else {
      int length = ((Object[]) data).length;
      RefList<int[]> childDims = RefArrays
          .stream(((Object[]) data)).map(x -> dims(x)).collect(RefCollectors.toList());
      int[] head = childDims.get(0);
      childDims.stream().forEach(d -> RefArrays.equals(head, d));
      return RefIntStream.concat(RefIntStream.of(length),
          RefArrays.stream(head)).toArray();
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
      return splitBuffer.stream().map(String::trim).filter(x -> !x.isEmpty()).map(VGG_WCT_Import::parse).toArray();
    } else {
      return Double.parseDouble(data);
    }
  }
}
