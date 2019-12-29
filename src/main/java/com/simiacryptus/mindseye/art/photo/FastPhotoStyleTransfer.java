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

package com.simiacryptus.mindseye.art.photo;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.RelativeAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.RefOperator;
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda;
import com.simiacryptus.mindseye.art.photo.topology.RadiusRasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import static com.simiacryptus.mindseye.lang.Layer.fromJson;
import static com.simiacryptus.util.JsonUtil.toJson;

/**
 * Implemented process detailed in:
 * A Closed-form Solution to Photorealistic Image Stylization
 * https://arxiv.org/pdf/1802.06474.pdf
 */
public class FastPhotoStyleTransfer extends ReferenceCountingBase implements Function<Tensor, UnaryOperator<Tensor>> {

  public final Layer encode_1;
  public final Layer decode_1;
  public final Layer encode_2;
  public final Layer decode_2;
  public final Layer encode_3;
  public final Layer decode_3;
  public final Layer encode_4;
  public final Layer decode_4;
  private boolean useCuda = true;
  private boolean smooth = true;
  private double lambda = 1e-4;
  private double epsilon = 1e-7;

  public FastPhotoStyleTransfer(Layer decode_1, Layer encode_1, @NotNull Layer decode_2, Layer encode_2, @NotNull Layer decode_3, Layer encode_3, @NotNull Layer decode_4, Layer encode_4) {
    this.encode_4 = encode_4;
    this.decode_4 = decode_4;
    this.encode_3 = encode_3;
    this.decode_3 = decode_3;
    this.encode_2 = encode_2;
    this.decode_2 = decode_2;
    this.encode_1 = encode_1;
    this.decode_1 = decode_1;
  }

  @Nonnull
  public static FastPhotoStyleTransfer fromZip(@Nonnull final ZipFile zipfile) {
    @Nonnull HashMap<CharSequence, byte[]> resources = ZipSerializable.extract(zipfile);
    return new FastPhotoStyleTransfer(
        fromJson(toJson(resources.get("decode_1.json")), resources),
        fromJson(toJson(resources.get("encode_1.json")), resources),
        fromJson(toJson(resources.get("decode_2.json")), resources),
        fromJson(toJson(resources.get("encode_2.json")), resources),
        fromJson(toJson(resources.get("decode_3.json")), resources),
        fromJson(toJson(resources.get("encode_3.json")), resources),
        fromJson(toJson(resources.get("decode_4.json")), resources),
        fromJson(toJson(resources.get("encode_4.json")), resources)
    );
  }

  public static Tensor transfer(Tensor contentImage, Tensor styleImage, Layer encode, Layer decode, double contentDensity, double styleDensity) {
    final Tensor encodedContent = encode.eval(contentImage).getData().get(0);
    final Tensor encodedStyle = encode.eval(styleImage).getData().get(0);
    final PipelineNetwork applicator = WCTUtil.applicator(encodedStyle, contentDensity, styleDensity);
    encodedStyle.freeRef();
    final Tensor encodedTransformed = applicator.eval(encodedContent).getData().get(0);
    encodedContent.freeRef();
    applicator.freeRef();
    final Tensor tensor = decode.eval(encodedTransformed, contentImage).getData().get(0);
    encodedTransformed.freeRef();
    return tensor;
  }

  @Override
  protected void _free() {
    encode_1.freeRef();
    decode_1.freeRef();
    encode_2.freeRef();
    decode_2.freeRef();
    encode_3.freeRef();
    decode_3.freeRef();
    encode_4.freeRef();
    decode_4.freeRef();
    super._free();
  }

  public void writeZip(@Nonnull File out, SerialPrecision precision) {
    try (@Nonnull ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(out))) {
      final HashMap<CharSequence, byte[]> resources = new HashMap<>();
      decode_1.writeZip(zipOutputStream, precision, resources, "decode_1.json");
      encode_1.writeZip(zipOutputStream, precision, resources, "encode_1.json");
      decode_2.writeZip(zipOutputStream, precision, resources, "decode_2.json");
      encode_2.writeZip(zipOutputStream, precision, resources, "encode_2.json");
      decode_3.writeZip(zipOutputStream, precision, resources, "decode_3.json");
      encode_3.writeZip(zipOutputStream, precision, resources, "encode_3.json");
      decode_4.writeZip(zipOutputStream, precision, resources, "decode_4.json");
      encode_4.writeZip(zipOutputStream, precision, resources, "encode_4.json");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public RefOperator<Tensor> apply(Tensor contentImage) {
    return new StyleOperator(contentImage);
  }

  public Tensor photoWCT(Tensor style, Tensor content) {
    return photoWCT(style, content, 1.0, 1.0);
  }

  @NotNull
  public Tensor photoWCT(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return photoWCT_1(style,
        photoWCT_2(style,
            photoWCT_3(style,
                photoWCT_4(style, content, contentDensity, styleDensity),
                contentDensity, styleDensity),
            contentDensity, styleDensity),
        contentDensity, styleDensity);
  }

  public @NotNull Tensor photoWCT_1(Tensor style, Tensor content) {
    return photoWCT_1(style, content, 1.0, 1.0);
  }

  public @NotNull Tensor photoWCT_1(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    final Tensor encodedContent = encode_1.eval(content).getData().get(0);
    final Tensor encodedStyle = encode_1.eval(style).getData().get(0);
    final PipelineNetwork applicator = WCTUtil.applicator(encodedStyle, contentDensity, styleDensity);
    final Tensor encodedTransformed = applicator.eval(encodedContent).getData().get(0);
    encodedContent.freeRef();
    applicator.freeRef();
    encodedStyle.freeRef();
    final Tensor tensor = decode_1.eval(encodedTransformed).getData().get(0);
    encodedTransformed.freeRef();
    return tensor;
  }

  @NotNull
  public Tensor photoWCT_2(Tensor style, Tensor content) {
    return photoWCT_2(style, content, 1.0, 1.0);
  }

  @NotNull
  public Tensor photoWCT_2(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return transfer(content, style, encode_2, decode_2, contentDensity, styleDensity);
  }

  @NotNull
  public Tensor photoWCT_3(Tensor style, Tensor content) {
    return photoWCT_3(style, content, 1.0, 1.0);
  }

  @NotNull
  public Tensor photoWCT_3(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return transfer(content, style, encode_3, decode_3, contentDensity, styleDensity);
  }

  public Tensor photoWCT_4(Tensor style, Tensor content) {
    return photoWCT_4(style, content, 1.0, 1.0);
  }

  public Tensor photoWCT_4(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return transfer(content, style, encode_4, decode_4, contentDensity, styleDensity);
  }

  public RefOperator<Tensor> photoSmooth(Tensor content) {
    if (isSmooth()) {
      RasterTopology topology = new RadiusRasterTopology(content.getDimensions(), RadiusRasterTopology.getRadius(1, 1), -1);
//      RasterAffinity affinity = new MattingAffinity(mask, topology);
      RasterAffinity affinity = new RelativeAffinity(content, topology);
      //RasterAffinity affinity = new GaussianAffinity(mask, 20, topology);
      //affinity = new TruncatedAffinity(affinity).setMin(1e-2);
      return (isUseCuda() ? new SmoothSolver_Cuda() : new SmoothSolver_EJML()).solve(topology, affinity, getLambda());
    } else return new NullOperator();
  }

  public boolean isSmooth() {
    return smooth;
  }

  public FastPhotoStyleTransfer setSmooth(boolean smooth) {
    this.smooth = smooth;
    return this;
  }

  public double getLambda() {
    return lambda;
  }

  public FastPhotoStyleTransfer setLambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  public double getEpsilon() {
    return epsilon;
  }

  public FastPhotoStyleTransfer setEpsilon(double epsilon) {
    this.epsilon = epsilon;
    return this;
  }

  public boolean isUseCuda() {
    return useCuda;
  }

  public FastPhotoStyleTransfer setUseCuda(boolean useCuda) {
    this.useCuda = useCuda;
    return this;
  }

  private static class NullOperator<T> extends ReferenceCountingBase implements RefOperator<T> {
    @Override
    public T apply(T tensor) {
      return tensor;
    }
  }

  private class StyleOperator extends ReferenceCountingBase implements RefOperator<Tensor> {
    final RefOperator<Tensor> photoSmooth;
    private final Tensor contentImage;

    public StyleOperator(Tensor contentImage) {
      this.contentImage = contentImage;
      photoSmooth = photoSmooth(contentImage);
    }

    @Override
    protected void _free() {
      contentImage.freeRef();
      photoSmooth.freeRef();
      super._free();
    }

    @Override
    public Tensor apply(Tensor styleImage) {
      final Tensor tensor = FastPhotoStyleTransfer.this.photoWCT(styleImage, contentImage);
      final Tensor tensor1 = photoSmooth.apply(tensor);
      tensor.freeRef();
      return tensor1;
    }
  }
}
