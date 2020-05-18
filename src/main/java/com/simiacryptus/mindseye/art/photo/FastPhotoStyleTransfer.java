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

import com.simiacryptus.mindseye.art.photo.affinity.ContextAffinity;
import com.simiacryptus.mindseye.art.photo.affinity.RelativeAffinity;
import com.simiacryptus.mindseye.art.photo.cuda.RefUnaryOperator;
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda;
import com.simiacryptus.mindseye.art.photo.topology.RadiusRasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.util.Util;

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

  /**
   * The Encode 1.
   */
  public final Layer encode_1;
  /**
   * The Decode 1.
   */
  public final Layer decode_1;
  /**
   * The Encode 2.
   */
  public final Layer encode_2;
  /**
   * The Decode 2.
   */
  @Nonnull
  public final Layer decode_2;
  /**
   * The Encode 3.
   */
  public final Layer encode_3;
  /**
   * The Decode 3.
   */
  @Nonnull
  public final Layer decode_3;
  /**
   * The Encode 4.
   */
  public final Layer encode_4;
  /**
   * The Decode 4.
   */
  @Nonnull
  public final Layer decode_4;
  private boolean useCuda = true;
  private boolean smooth = true;
  private double lambda = 1e-4;
  private double epsilon = 1e-7;

  /**
   * Instantiates a new Fast photo style transfer.
   *
   * @param decode_1 the decode 1
   * @param encode_1 the encode 1
   * @param decode_2 the decode 2
   * @param encode_2 the encode 2
   * @param decode_3 the decode 3
   * @param encode_3 the encode 3
   * @param decode_4 the decode 4
   * @param encode_4 the encode 4
   */
  public FastPhotoStyleTransfer(Layer decode_1, Layer encode_1, @Nonnull Layer decode_2, Layer encode_2,
                                @Nonnull Layer decode_3, Layer encode_3, @Nonnull Layer decode_4, Layer encode_4) {
    this.encode_4 = encode_4;
    this.decode_4 = decode_4;
    this.encode_3 = encode_3;
    this.decode_3 = decode_3;
    this.encode_2 = encode_2;
    this.decode_2 = decode_2;
    this.encode_1 = encode_1;
    this.decode_1 = decode_1;
  }

  /**
   * Gets epsilon.
   *
   * @return the epsilon
   */
  public double getEpsilon() {
    return epsilon;
  }

  /**
   * Sets epsilon.
   *
   * @param epsilon the epsilon
   */
  public void setEpsilon(double epsilon) {
    this.epsilon = epsilon;
  }

  /**
   * Gets lambda.
   *
   * @return the lambda
   */
  public double getLambda() {
    return lambda;
  }

  /**
   * Sets lambda.
   *
   * @param lambda the lambda
   */
  public void setLambda(double lambda) {
    this.lambda = lambda;
  }

  /**
   * Is smooth boolean.
   *
   * @return the boolean
   */
  public boolean isSmooth() {
    return smooth;
  }

  /**
   * Sets smooth.
   *
   * @param smooth the smooth
   */
  public void setSmooth(boolean smooth) {
    this.smooth = smooth;
  }

  /**
   * Is use cuda boolean.
   *
   * @return the boolean
   */
  public boolean isUseCuda() {
    return useCuda;
  }

  /**
   * Sets use cuda.
   *
   * @param useCuda the use cuda
   * @return the use cuda
   */
  @Nonnull
  public FastPhotoStyleTransfer setUseCuda(boolean useCuda) {
    this.useCuda = useCuda;
    return this;
  }

  /**
   * From zip fast photo style transfer.
   *
   * @param zipfile the zipfile
   * @return the fast photo style transfer
   */
  @Nonnull
  public static FastPhotoStyleTransfer fromZip(@Nonnull final ZipFile zipfile) {
    @Nonnull
    HashMap<CharSequence, byte[]> resources = ZipSerializable.extract(zipfile);
    return new FastPhotoStyleTransfer(fromJson(toJson(resources.get("decode_1.json")), resources),
        fromJson(toJson(resources.get("encode_1.json")), resources),
        fromJson(toJson(resources.get("decode_2.json")), resources),
        fromJson(toJson(resources.get("encode_2.json")), resources),
        fromJson(toJson(resources.get("decode_3.json")), resources),
        fromJson(toJson(resources.get("encode_3.json")), resources),
        fromJson(toJson(resources.get("decode_4.json")), resources),
        fromJson(toJson(resources.get("encode_4.json")), resources));
  }

  /**
   * Transfer tensor.
   *
   * @param contentImage   the content image
   * @param styleImage     the style image
   * @param encode         the encode
   * @param decode         the decode
   * @param contentDensity the content density
   * @param styleDensity   the style density
   * @return the tensor
   */
  @Nonnull
  public static Tensor transfer(Tensor contentImage, Tensor styleImage, @Nonnull Layer encode, @Nonnull Layer decode,
                                double contentDensity, double styleDensity) {
    final Tensor encodedContent = Result.getData0(encode.eval(contentImage.addRef()));
    final Tensor encodedStyle = Result.getData0(encode.eval(styleImage));
    encode.freeRef();
    final PipelineNetwork applicator = WCTUtil.applicator(encodedStyle, contentDensity, styleDensity);
    final Tensor encodedTransformed = Result.getData0(applicator.eval(encodedContent));
    applicator.freeRef();
    final Tensor tensor = Result.getData0(decode.eval(encodedTransformed, contentImage));
    decode.freeRef();
    return tensor;
  }

  public void _free() {
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

  /**
   * Write zip.
   *
   * @param out       the out
   * @param precision the precision
   */
  public void writeZip(@Nonnull File out, SerialPrecision precision) {
    try (@Nonnull
         ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(out))) {
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
      throw Util.throwException(e);
    }
  }

  @Nonnull
  public RefUnaryOperator<Tensor> apply(@Nonnull Tensor contentImage) {
    return new StyleUnaryOperator(contentImage, FastPhotoStyleTransfer.this);
  }

  /**
   * Photo wct tensor.
   *
   * @param style   the style
   * @param content the content
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT(Tensor style, Tensor content) {
    return photoWCT(style, content, 1.0, 1.0);
  }

  /**
   * Photo wct tensor.
   *
   * @param style          the style
   * @param content        the content
   * @param contentDensity the content density
   * @param styleDensity   the style density
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    Tensor wct1 = photoWCT_1(style.addRef(),
        photoWCT_2(style.addRef(),
            photoWCT_3(style.addRef(), photoWCT_4(style.addRef(), content, contentDensity, styleDensity), contentDensity, styleDensity),
            contentDensity, styleDensity),
        contentDensity, styleDensity);
    style.freeRef();
    return wct1;
  }

  /**
   * Photo wct 1 tensor.
   *
   * @param style   the style
   * @param content the content
   * @return the tensor
   */
  public @Nonnull
  Tensor photoWCT_1(Tensor style, Tensor content) {
    return photoWCT_1(style, content, 1.0, 1.0);
  }

  /**
   * Photo wct 1 tensor.
   *
   * @param style          the style
   * @param content        the content
   * @param contentDensity the content density
   * @param styleDensity   the style density
   * @return the tensor
   */
  public @Nonnull
  Tensor photoWCT_1(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    final Tensor encodedContent = Result.getData0(encode_1.eval(content));
    final Tensor encodedStyle = Result.getData0(encode_1.eval(style));
    final PipelineNetwork applicator = WCTUtil.applicator(encodedStyle, contentDensity, styleDensity);
    final Tensor encodedTransformed = Result.getData0(applicator.eval(encodedContent));
    applicator.freeRef();
    return Result.getData0(decode_1.eval(encodedTransformed));
  }

  /**
   * Photo wct 2 tensor.
   *
   * @param style   the style
   * @param content the content
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT_2(Tensor style, Tensor content) {
    return photoWCT_2(style, content, 1.0, 1.0);
  }

  /**
   * Photo wct 2 tensor.
   *
   * @param style          the style
   * @param content        the content
   * @param contentDensity the content density
   * @param styleDensity   the style density
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT_2(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return transfer(content, style, encode_2.addRef(), decode_2.addRef(), contentDensity, styleDensity);
  }

  /**
   * Photo wct 3 tensor.
   *
   * @param style   the style
   * @param content the content
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT_3(Tensor style, Tensor content) {
    return photoWCT_3(style, content, 1.0, 1.0);
  }

  /**
   * Photo wct 3 tensor.
   *
   * @param style          the style
   * @param content        the content
   * @param contentDensity the content density
   * @param styleDensity   the style density
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT_3(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return transfer(content, style, encode_3.addRef(), decode_3.addRef(), contentDensity, styleDensity);
  }

  /**
   * Photo wct 4 tensor.
   *
   * @param style   the style
   * @param content the content
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT_4(Tensor style, Tensor content) {
    return photoWCT_4(style, content, 1.0, 1.0);
  }

  /**
   * Photo wct 4 tensor.
   *
   * @param style          the style
   * @param content        the content
   * @param contentDensity the content density
   * @param styleDensity   the style density
   * @return the tensor
   */
  @Nonnull
  public Tensor photoWCT_4(Tensor style, Tensor content, double contentDensity, double styleDensity) {
    return transfer(content, style, encode_4.addRef(), decode_4.addRef(), contentDensity, styleDensity);
  }

  /**
   * Photo smooth ref unary operator.
   *
   * @param content the content
   * @return the ref unary operator
   */
  @Nonnull
  public RefUnaryOperator<Tensor> photoSmooth(@Nonnull Tensor content) {
    if (isSmooth()) {
      RasterTopology topology = new RadiusRasterTopology(content.getDimensions(), RadiusRasterTopology.getRadius(1, 1),
          -1);
      //      RasterAffinity affinity = new MattingAffinity(mask, topology);
      ContextAffinity affinity = new RelativeAffinity(content, topology.addRef());
      //RasterAffinity affinity = new GaussianAffinity(mask, 20, topology);
      //affinity = new TruncatedAffinity(affinity).setMin(1e-2);
      return (isUseCuda() ? new SmoothSolver_Cuda() : new SmoothSolver_EJML()).solve(topology, affinity, getLambda());
    } else
      content.freeRef();
    return new NullUnaryOperator();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  FastPhotoStyleTransfer addRef() {
    return (FastPhotoStyleTransfer) super.addRef();
  }

  private static class NullUnaryOperator<T> extends ReferenceCountingBase implements RefUnaryOperator<T> {

    @Override
    public T apply(T tensor) {
      return tensor;
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    NullUnaryOperator<T> addRef() {
      return (NullUnaryOperator<T>) super.addRef();
    }
  }

  private static class StyleUnaryOperator extends ReferenceCountingBase implements RefUnaryOperator<Tensor> {
    /**
     * The Photo smooth.
     */
    @Nonnull
    final RefUnaryOperator<Tensor> photoSmooth;
    @Nonnull
    private final Tensor contentImage;
    @Nonnull
    private final FastPhotoStyleTransfer parent;

    /**
     * Instantiates a new Style unary operator.
     *
     * @param contentImage the content image
     * @param parent       the parent
     */
    public StyleUnaryOperator(@Nonnull Tensor contentImage, @Nonnull FastPhotoStyleTransfer parent) {
      this.parent = parent;
      this.contentImage = contentImage;
      photoSmooth = this.parent.photoSmooth(this.contentImage);
    }

    public void _free() {
      contentImage.freeRef();
      photoSmooth.freeRef();
      parent.freeRef();
      super._free();
    }

    @Override
    public Tensor apply(Tensor styleImage) {
      return photoSmooth.apply(parent.photoWCT(styleImage, contentImage.addRef()));
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    StyleUnaryOperator addRef() {
      return (StyleUnaryOperator) super.addRef();
    }
  }
}
