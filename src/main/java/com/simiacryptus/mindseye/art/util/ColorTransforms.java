package com.simiacryptus.mindseye.art.util;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RecycleBin;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.simiacryptus.ref.lang.RecycleBin.DOUBLES;

public class ColorTransforms {

  public interface ColorTransform {
    void transform(double[] imagePixel, double[] resultPixel);
  }

  public static Tensor colorTransform(Tensor image, ColorTransform fn) {
    @Nonnull int[] imageDimensions = image.getDimensions();
    int bands = 3;
    if (bands != imageDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", bands, imageDimensions[2]));
    Tensor result = image.copy();
    int width = imageDimensions[0];
    int height = imageDimensions[1];

    IntStream.range(0, width).parallel().forEach(x -> {
      double[] imagePixel = DOUBLES.obtain(bands);
      double[] resultPixel = DOUBLES.obtain(bands);
      IntStream.range(0, height).forEach(y -> {
        image.getPixel(x, y, imagePixel);
        fn.transform(imagePixel, resultPixel);
        result.setPixel(x, y, resultPixel);
      });
      DOUBLES.recycle(imagePixel, bands);
      DOUBLES.recycle(resultPixel, bands);
    });

    image.freeRef();
    return result;
  }

  public static Tensor hsl2rgb(Tensor image) {
    return colorTransform(image, ColorTransforms::hsl2rgb);
  }

  public static Tensor rgb2hsl(Tensor image) {
    return colorTransform(image, ColorTransforms::rgb2hsl);
  }

  public static Tensor hsv2rgb(Tensor image) {
    return colorTransform(image, ColorTransforms::hsv2rgb);
  }

  public static Tensor rgb2hsv(Tensor image) {
    return colorTransform(image, ColorTransforms::rgb2hsv);
  }

  public static Tensor xyz2rgb(Tensor image) {
    return colorTransform(image, ColorTransforms::xyz2rgb);
  }

  public static Tensor rgb2xyz(Tensor image) {
    return colorTransform(image, ColorTransforms::rgb2xyz);
  }

  public static Tensor lab2rgb(Tensor image) {
    return colorTransform(image, ColorTransforms::lab2rgb);
  }

  public static Tensor rgb2lab(Tensor image) {
    return colorTransform(image, ColorTransforms::rgb2lab);
  }

  public static void hsl2hsv(double[] imagePixel, double[] resultPixel) {
    double h = imagePixel[0];
    double sl = imagePixel[1] / 100;
    double l = imagePixel[2] / 100;
    double v = l + sl * Math.min(l, 1 - l);
    double sv = v == 0.0 ? 0 : (2 * (1 - l / v));
    resultPixel[0] = h;
    resultPixel[1] = sv * 100;
    resultPixel[2] = l * 100;
  }

  public static void hsv2hsl(double[] imagePixel, double[] resultPixel) {
    double h = imagePixel[0];
    double sv = imagePixel[1] / 100;
    double v = imagePixel[2] / 100;
    double l = v * (1 - sv / 2);
    double sl = (Math.min(l, 1 - l) == 0.0) ? 0 : ((v - l) / Math.min(l, 1 - l));
    resultPixel[0] = h;
    resultPixel[1] = sl * 100;
    resultPixel[2] = l * 100;
  }

  public static void rgb2hsv(double[] imagePixel, double[] resultPixel) {
    rgb2hsl(imagePixel, resultPixel);
    hsl2hsv(resultPixel, resultPixel);
  }

  public static void hsv2rgb(double[] imagePixel, double[] resultPixel) {
    hsv2hsl(imagePixel, resultPixel);
    hsl2rgb(resultPixel, resultPixel);
  }

  public static void hsl2rgb(double[] imagePixel, double[] resultPixel) {
    double h = imagePixel[0];
    double s = Math.min(Math.max(imagePixel[1] / 100, 0.0), 1.0);
    double l = Math.min(Math.max(imagePixel[2] / 100, 0.0), 1.0);
    double r;
    double g;
    double b;

    while (h < 0) {
      h += 360;
    }
    h = h % 360;

    double a = s * Math.min(l, 1 - l);
    double k;
    k = (h / 30) % 12;
    r = l - a * Math.max(-1, Math.min(Math.min(k - 3, 9 - k), 1));
    k = (8 + h / 30) % 12;
    g = l - a * Math.max(-1, Math.min(Math.min(k - 3, 9 - k), 1));
    k = (4 + h / 30) % 12;
    b = l - a * Math.max(-1, Math.min(Math.min(k - 3, 9 - k), 1));

    resultPixel[0] = r * 256;
    resultPixel[1] = g * 256;
    resultPixel[2] = b * 256;
  }

  public static void rgb2hsl(double[] imagePixel, double[] resultPixel) {
    double r = Math.min(Math.max(imagePixel[0] / 256, 0.0), 1.0);
    double g = Math.min(Math.max(imagePixel[1] / 256, 0.0), 1.0);
    double b = Math.min(Math.max(imagePixel[2] / 256, 0.0), 1.0);
    double cmin = Math.min(Math.min(r, g), b);
    double cmax = Math.max(Math.max(r, g), b);
    double chroma = cmax - cmin;
    double h;
    double s;
    double l;

    l = cmax - chroma / 2;
    s = (chroma <= 0.0 || chroma >= 1.0) ? 0 : (chroma / (1 - Math.abs(2 * l - 1)));

    if (chroma == 0) {
      h = 0;
    } else if (cmax == r) {
      h = (g - b) / chroma;
    } else if (cmax == g) {
      h = ((b - r) / chroma) + 2;
    } else {
      h = ((r - g) / chroma) + 4;
    }
    h *= 60;
    if (h < 0) h += 360;
    h %= 360;

    resultPixel[0] = h;
    resultPixel[1] = s * 100;
    resultPixel[2] = l * 100;
  }

  public static void lab2rgb(double[] imagePixel, double[] resultPixel) {
    lab2xyz(imagePixel, resultPixel);
    xyz2rgb(resultPixel, resultPixel);
  }

  public static void xyz2rgb(double[] imagePixel, double[] resultPixel) {
    double x = imagePixel[0];
    double y = imagePixel[1];
    double z = imagePixel[2];

    double red;
    double blue;
    double green;

    red = ((x * 3.24048) + (y * -1.53715) + (z * -0.498536));
    green = ((x * -0.969255) + (y * 1.87599) + (z * 0.0415559));
    blue = ((x * 0.0556466) + (y * -0.204041) + (z * 1.05731));

    red /= 100.0;
    green /= 100.0;
    blue /= 100.0;


    if (red > (.04045f / 12.92f)) {
      red = Math.pow(red, 1 / 2.4) * 1.055 - .055;
    } else {
      red = red * 12.92f;
    }

    if (green > (.04045f / 12.92f)) {
      green = Math.pow(green, 1 / 2.4) * 1.055 - .055;
    } else {
      green = green * 12.92f;
    }

    if (blue > (.04045f / 12.92f)) {
      blue = Math.pow(blue, 1 / 2.4) * 1.055 - .055;
    } else {
      blue = blue * 12.92f;
    }

    resultPixel[0] = red * 256;
    resultPixel[1] = green * 256;
    resultPixel[2] = blue * 256;
  }

  public static void lab2xyz(double[] imagePixel, double[] resultPixel) {
    double l = imagePixel[0];
    double a = imagePixel[1];
    double b = imagePixel[2];
    double x;
    double y;
    double z;

    l += 16;
    x = 0.00862069 * l + 0.002 * a;
    y = 0.00862069 * l;
    z = 0.00862069 * l - 0.005 * b;

    if (x > 0.20689271) {
      x = Math.pow(x, 3);
    } else {
      x = (x - (16.0f / 116.0f)) / 7.787f;
    }

    if (y > 0.20689271) {
      y = Math.pow(y, 3);
    } else {
      y = (y - (16.0f / 116.0f)) / 7.787f;
    }

    if (z > 0.20689271) {
      z = Math.pow(z, 3);
    } else {
      z = (z - (16.0f / 116.0f)) / 7.787f;
    }

    resultPixel[0] = x * 95.047;
    resultPixel[1] = y * 100.0;
    resultPixel[2] = z * 108.883;
  }

  public static void rgb2xyz(double[] imagePixel, double[] resultPixel) {
    double red = Math.min(Math.max(imagePixel[0] / 256, 0.0), 1.0);
    double green = Math.min(Math.max(imagePixel[1] / 256, 0.0), 1.0);
    double blue = Math.min(Math.max(imagePixel[2] / 256, 0.0), 1.0);

    if (red > .04045f) {
      red = Math.pow((red + .055) / 1.055, 2.4);
    } else {
      red = red / 12.92f;
    }

    if (green > .04045f) {
      green = Math.pow((green + .055) / 1.055, 2.4);
    } else {
      green = green / 12.92f;
    }

    if (blue > .04045f) {
      blue = Math.pow((blue + .055) / 1.055, 2.4);
    } else {
      blue = blue / 12.92f;
    }

    red *= 100.0;
    green *= 100.0;
    blue *= 100.0;

    double x = ((red * .412453) + (green * .357580) + (blue * .180423));
    double y = ((red * .212671) + (green * .715160) + (blue * .072169));
    double z = ((red * .019334) + (green * .119193) + (blue * .950227));

    resultPixel[0] = x;
    resultPixel[1] = y;
    resultPixel[2] = z;
  }

  public static void rgb2lab(double[] imagePixel, double[] resultPixel) {
    rgb2xyz(imagePixel, resultPixel);
    xyz2lab(resultPixel, resultPixel);
  }

  public static void xyz2lab(double[] imagePixel, double[] resultPixel) {
    double x = imagePixel[0] / 95.047f;
    double y = imagePixel[1] / 100.0f;
    double z = imagePixel[2] / 108.883f;
    double l;
    double a;
    double b;

    if (x > .008856f) {
      x = Math.pow(x, (1.0 / 3.0));
    } else {
      x = (x * 7.787f) + (16.0f / 116.0f);
    }

    if (y > .008856f) {
      y = Math.pow(y, 1.0 / 3.0);
    } else {
      y = (y * 7.787f) + (16.0f / 116.0f);
    }

    if (z > .008856f) {
      z = Math.pow(z, 1.0 / 3.0);
    } else {
      z = (z * 7.787f) + (16.0f / 116.0f);
    }

    l = (116.0f * y) - 16.0f;
    a = 500.0f * (x - y);
    b = 200.0f * (y - z);

    resultPixel[0] = l;
    resultPixel[1] = a;
    resultPixel[2] = b;
  }

  public static Tensor colorTransfer(Tensor reference, Tensor image) {
    return colorTransfer_random(reference, image, 30);
  }

  public static Tensor colorTransfer_primaries(Tensor reference, Tensor image) {
    @Nonnull int[] imageDimensions = image.getDimensions();
    int bands = imageDimensions[2];
    @Nonnull double[] vectorData = new double[bands];
    for (int i = 0; i < bands; i++) {
      Arrays.fill(vectorData, 0.0);
      vectorData[i] = 1.0;
      double adjustmentSum = colorTransfer(reference, image, vectorData, (dotProduct, targetValue) -> targetValue);
      System.out.printf("adjustment[%d]=%.3f%n", i, adjustmentSum / (imageDimensions[0] * imageDimensions[1]));
    }
    return image;
  }

  public static Tensor colorTransfer_random(Tensor reference, Tensor image, int iterations) {
    return colorTransfer_random(reference, image, iterations, (dotProduct, targetValue) -> targetValue);
  }

  public static Tensor colorTransfer_random_geometricDamping(Tensor reference, Tensor image, int iterations) {
    return colorTransfer_random_geometricDamping(reference, image, iterations, 1.0 / 2.0);
  }

  @NotNull
  public static Tensor colorTransfer_random_geometricDamping(Tensor reference, Tensor image, int iterations, double geometricDappening) {
    return colorTransfer_random(reference, image, iterations, (dotProduct, targetValue) -> {
      if (Math.abs(dotProduct) < 1e-6) {
        return dotProduct;
      } else {
        return dotProduct * Math.pow(targetValue / dotProduct, geometricDappening);
      }
    });
  }

  public static Tensor colorTransfer_random_arithmeticDamping(Tensor reference, Tensor image, int iterations) {
    return colorTransfer_random_arithmeticDamping(reference, image, iterations, 1.0 / 2.0);
  }

  @NotNull
  public static Tensor colorTransfer_random_arithmeticDamping(Tensor reference, Tensor image, int iterations, double arithmeticDappening) {
    return colorTransfer_random(reference, image, iterations, (dotProduct, targetValue) -> {
      return dotProduct + (targetValue - dotProduct) * arithmeticDappening;
    });
  }

  @NotNull
  public static Tensor colorTransfer_random(Tensor reference, Tensor image, int iterations, ScalarRemappingFunction fn) {
    @Nonnull int[] imageDimensions = image.getDimensions();
    Tensor vector = new Tensor(new int[]{1, 1, imageDimensions[2]});
    for (int i = 0; i < iterations; i++) {
      vector.randomize(1.0);
      vector.scaleInPlace(1.0 / vector.sumSq());
      @Nonnull double[] vectorData = vector.getData();
      double adjustmentSum = colorTransfer(reference, image, vectorData, fn);
      System.out.printf("adjustment[%d]=%.3f%n", i, adjustmentSum / (imageDimensions[0] * imageDimensions[1]));
    }
    vector.freeRef();
    return image;
  }

  public interface ScalarRemappingFunction {
    double apply(double value, double targetValue);
  }

  public static double colorTransfer(Tensor reference, Tensor image, @Nonnull double[] colorVector, ScalarRemappingFunction fn) {
    @Nonnull int[] referenceDimensions = reference.getDimensions();
    @Nonnull int[] imageDimensions = image.getDimensions();
    int bands = referenceDimensions[2];
    if (bands != imageDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", bands, imageDimensions[2]));
    int refWidth = referenceDimensions[0];
    int refHeight = referenceDimensions[1];
    int imgWidth = imageDimensions[0];
    int imgHeight = imageDimensions[1];

    double[] refHistogram = DOUBLES.obtain(refWidth * refHeight);
    double[] imgHistogram = DOUBLES.obtain(imgWidth * imgHeight);

    IntStream.range(0, refWidth).parallel().forEach(x -> {
      double[] pixel = DOUBLES.obtain(bands);
      int offset = refHeight * x;
      IntStream.range(0, refHeight).forEach(y -> {
        reference.getPixel(x, y, pixel);
        double dotProduct = 0;
        for (int c = 0; c < bands; c++) {
          dotProduct += colorVector[c] * pixel[c];
        }
        refHistogram[y + offset] = dotProduct;
      });
      DOUBLES.recycle(pixel, bands);
    });
    Arrays.sort(refHistogram);

    IntStream.range(0, imgWidth).parallel().forEach(x -> {
      double[] pixel = DOUBLES.obtain(bands);
      int offset = imgHeight * x;
      IntStream.range(0, imgHeight).forEach(y -> {
        image.getPixel(x, y, pixel);
        double dotProduct = 0;
        for (int c = 0; c < bands; c++) {
          dotProduct += colorVector[c] * pixel[c];
        }
        imgHistogram[y + offset] = dotProduct;
      });
      DOUBLES.recycle(pixel, bands);
    });
    Arrays.sort(imgHistogram);

    double adj = (double) refHistogram.length / imgHistogram.length;
    double adjustmentSum = IntStream.range(0, imgWidth).parallel().mapToDouble(x -> {
      double[] pixel = DOUBLES.obtain(bands);
      double sum = IntStream.range(0, imgHeight).mapToDouble(y -> {
        image.getPixel(x, y, pixel);
        double dotProduct = 0;
        for (int c = 0; c < bands; c++) {
          dotProduct += colorVector[c] * pixel[c];
        }
        int imgIndex = Arrays.binarySearch(imgHistogram, dotProduct);
        if (imgIndex < 0) imgIndex = -(imgIndex + 1);
        int targetIndex = (int) (imgIndex * adj);
        double targetValue = refHistogram[targetIndex];
        targetValue = fn.apply(dotProduct, targetValue);
        double adjustment = targetValue - dotProduct;
        for (int c = 0; c < bands; c++) {
          pixel[c] = pixel[c] + colorVector[c] * adjustment;
        }
        image.setPixel(x, y, pixel);
        return Math.abs(adjustment);
      }).sum();
      DOUBLES.recycle(pixel, bands);
      return sum;
    }).sum();

    DOUBLES.recycle(refHistogram, refWidth * refHeight);
    DOUBLES.recycle(imgHistogram, imgWidth * imgHeight);

    return adjustmentSum;
  }

}
