package app.pivo.android.basicsdkdemo.tflite;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import app.pivo.android.basicsdkdemo.env.Utils;

public class FeatureExtract {
    private int INPUT_SIZE;
    private int OUTPUT_SHAPE;
    private Vector<String> labels = new Vector<>();

    private int NUM_THREADS = 4;
    private Interpreter TFLITE;

    public FeatureExtract(final AssetManager assetManager,
                          final String modelFilename,
                          final int input_size,
                          final int output_shape) {

        INPUT_SIZE = input_size;
        OUTPUT_SHAPE = output_shape;

        try {
            Interpreter.Options options = new Interpreter.Options();
            CompatibilityList compatList = new CompatibilityList();

            if(compatList.isDelegateSupportedOnThisDevice()){
                // if the device has a supported GPU, add the GPU delegate
                GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(NUM_THREADS);
            }

            TFLITE = new Interpreter(Utils.loadModelFile(assetManager, modelFilename), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    public float[] getFeature(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

        Map<Integer, Object> outputMap = new HashMap<>();

        outputMap.put(0, new float[1][OUTPUT_SHAPE]);
        Object[] inputArray = {byteBuffer};

        TFLITE.runForMultipleInputsOutputs(inputArray, outputMap);

        float[] out = ((float[][])outputMap.get(0))[0];

        return out;
    }
}
