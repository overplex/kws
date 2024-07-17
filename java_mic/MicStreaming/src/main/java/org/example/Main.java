package org.example;

import org.apache.commons.cli.*;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.config.Configurator;

import javax.sound.sampled.*;
import java.util.ArrayList;
import java.util.Arrays;

import static org.example.TfKws.NOT_KW;
import static org.example.TfKws.SILENCE;

public class Main {

    private static final int CHANNELS = 1;
    private static final int BITS_PER_SAMPLE = 16;
    private static final boolean SIGNED = true;
    private static final boolean BIG_ENDIAN = false;


    public static void main(String[] args) {
        CommandLineParser parser = new DefaultParser();
        Options options = new Options();
        CommandLine cmd = null;

        try {
            options.addOption(new Option("l", "list-devices", false,
                    "show list of audio devices and exit"));
            options.addOption("m", "model", true,
                    "path of tensorflow model directory");
            options.addOption("k", "keywords", true,
                    "keywords separated by comma");
            options.addOption("i", "input-device", true,
                    "input device name");
            options.addOption("r", "sample-rate", true,
                    "input sample rate");
            options.addOption("b", "block-len-ms", true,
                    "input block (window stride) length (ms)");
            options.addOption("s", "score-strategy", true,
                    "score strategy, choose between 'smoothed_confidence' or 'hit_ratio' (default)");
            options.addOption(null, "score-threshold", true,
                    "score threshold, if not specified, this is automatically determined by strategy");
            options.addOption(null, "hit-threshold", true,
                    "hit threshold");
            options.addOption(null, "tailroom-ms", true,
                    "tail room in ms");
            options.addOption(null, "add-softmax", false,
                    "whether add softmax layer to output");
            options.addOption(null, "silence-on", false,
                    "turn on silence detection");
            options.addOption(null, "delay-trigger", false,
                    "only trigger after utterance end");
            options.addOption(null, "lookahead-ms", true,
                    "silence lookahead window to prevent kw in sentence, 0 to turn off (default 0 ms)");
            options.addOption(null, "min-kw-ms", true,
                    "minimum kw duration (default 100 ms)");
            options.addOption(null, "max-kw", true,
                    "max number of kw in one utterance");
            options.addOption("v", "verbose", true,
                    "verbose level: 0 - quiet, 1 - info, 2 - debug, 3 - verbose");
            options.addOption("h", "help", false,
                    "show help");

            cmd = parser.parse(options, args);

            if (cmd.hasOption("l")) {
                showAllMicrophones();
                System.exit(0);
            }

            if (cmd.hasOption("h")) {
                printHelp(options);
                System.exit(0);
            }

            if (cmd.hasOption("v")) {
                switch (Integer.parseInt(cmd.getOptionValue("v"))) {
                    case 1 -> Configurator.setRootLevel(Level.INFO);
                    case 2 -> Configurator.setRootLevel(Level.DEBUG);
                    case 3 -> Configurator.setRootLevel(Level.TRACE);
                    default -> Configurator.setRootLevel(Level.ERROR);
                }
            }
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            printHelp(options);
            System.exit(1);
        }

        String keywords = cmd.getOptionValue("k");
        String modelPath = cmd.getOptionValue("m");

        if (keywords == null || modelPath == null) {
            System.err.println("Error: Missing required options: --model and --keywords");
            System.exit(1);
        }

        String inputDeviceName = cmd.getOptionValue("i");
        String scoreStrategy = cmd.getOptionValue("s", "hit_ratio");
        int maxKw = Integer.parseInt(cmd.getOptionValue("max-kw", "1"));
        int blockLenMs = Integer.parseInt(cmd.getOptionValue("b", "20"));
        int sampleRate = Integer.parseInt(cmd.getOptionValue("r", "16000"));
        int minKwMs = Integer.parseInt(cmd.getOptionValue("min-kw-ms", "100"));
        int tailroomMs = Integer.parseInt(cmd.getOptionValue("tailroom-ms", "100"));
        int lookaheadMs = Integer.parseInt(cmd.getOptionValue("lookahead-ms", "0"));
        float hitThreshold = Float.parseFloat(cmd.getOptionValue("hit-threshold", "7"));
        float scoreThreshold = cmd.hasOption("score-threshold")
                ? Float.parseFloat(cmd.getOptionValue("score-threshold"))
                : (scoreStrategy.equals("hit_ratio") ? 0.01f : 0.8f);
        boolean addSoftmax = cmd.hasOption("add-softmax");
        boolean silenceOn = cmd.hasOption("silence-on");
        boolean delayTrigger = cmd.hasOption("delay-trigger");

        ArrayList<String> keywordsList = new ArrayList<>();
        keywordsList.add(SILENCE);
        keywordsList.add(NOT_KW);
        keywordsList.addAll(Arrays.asList(keywords.split(",")));

        TfKws kws = new TfKws(modelPath, keywordsList.toArray(new String[0]));
        kws.setScoreThreshold(scoreThreshold);
        kws.setScoreStrategy(scoreStrategy);
        kws.setHitThreshold(hitThreshold);
        kws.setLookaheadMs(lookaheadMs);
        kws.setTailRoomMs(tailroomMs);
        kws.setBlockMs(blockLenMs);
        kws.setMinKwMs(minKwMs);
        kws.setMaxKwCnt(maxKw);

        if (addSoftmax) {
            kws.addSoftmaxLayerToOutput();
        }

        if (silenceOn) {
            kws.disableSilenceOff();
        }

        if (delayTrigger) {
            kws.disableImmediateTrigger();
        }

        openStream(kws, sampleRate, inputDeviceName);
    }

    private static void printHelp(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp("MicStreaming.jar -k sheila,marvin -m crnn_state_sheila_marvin_dir", options);
    }

    private static void showAllMicrophones() {
        for (Mixer.Info mixerInfo : AudioSystem.getMixerInfo()) {
            Mixer mixer = AudioSystem.getMixer(mixerInfo);
            for (Line.Info lineInfo : mixer.getTargetLineInfo()) {
                if (lineInfo.getLineClass().equals(TargetDataLine.class)) {
                    System.out.println(mixerInfo.getName());
                }
            }
        }
    }

    private static TargetDataLine findMicrophone(String name, AudioFormat format) throws LineUnavailableException {
        if (name == null) {
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
            return (TargetDataLine) AudioSystem.getLine(info);
        }

        for (Mixer.Info mixerInfo : AudioSystem.getMixerInfo()) {
            Mixer mixer = AudioSystem.getMixer(mixerInfo);
            for (Line.Info lineInfo : mixer.getTargetLineInfo()) {
                if (lineInfo.getLineClass().equals(TargetDataLine.class) &&
                        mixerInfo.getName().contains(name)) {
                    return (TargetDataLine) mixer.getLine(lineInfo);
                }
            }
        }

        throw new RuntimeException("Failed to find microphone \"" + name + "\"");
    }

    private static void openStream(TfKws kws, int sampleRate, String inputDeviceName) {
        int floatBufferSize = (int) kws.getBlockSize();
        int byteBufferSize = floatBufferSize * 2;

        AudioFormat format = new AudioFormat(sampleRate, BITS_PER_SAMPLE, CHANNELS, SIGNED, BIG_ENDIAN);

        try (TargetDataLine microphone = findMicrophone(inputDeviceName, format)) {
            microphone.open(format);

            int readBytes;
            String[] keywords;
            byte[] byteBuffer = new byte[byteBufferSize];
            float[] floatBuffer = new float[floatBufferSize];
            microphone.start();

            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                microphone.close();
                kws.close();
                System.out.println("Closed");
            }));

            try {
                System.out.println("Listening...");
                while (microphone.isOpen()) {
                    readBytes = microphone.read(byteBuffer, 0, byteBufferSize);
                    if (readBytes == byteBufferSize) {

                        // We need to feed in float values between -1.0f and 1.0f, so divide the
                        // signed 16-bit inputs.
                        for (int i = 0, j = 0; i < byteBufferSize; i += 2, j++) {
                            floatBuffer[j] = ((byteBuffer[i] & 0x00ff) | (byteBuffer[i + 1] << 8)) / 32767.0f;
                        }

                        keywords = kws.process(floatBuffer);

                        if (keywords.length > 0) {
                            System.out.println("API returned: " + String.join(", ", keywords));
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } catch (LineUnavailableException e) {
            e.printStackTrace();
        }
    }
}