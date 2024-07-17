package org.example;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.SignatureDef;
import org.tensorflow.proto.TensorInfo;
import org.tensorflow.types.TFloat32;

import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class TfKws {
    public static final String SILENCE = "silence";
    public static final String NOT_KW = "not_kw";

    private final Logger logger = LogManager.getLogger(getClass().getSimpleName());
    private final List<String> labels;

    private final long blockSize;

    private final int notKwId;
    private final int silenceId;
    private final int tailThreshold;
    private final int countKeywords;

    private final Map<String, Integer> utteranceHits = new HashMap<>();
    private final Map<String, Float> utteranceScores = new HashMap<>();
    private final List<String> alreadyTriggered = new ArrayList<>();
    private final int[] keywordsIdx;


    private final CircularFifoQueue<Integer> labelRing;
    private final CircularFifoQueue<float[]> scoreWindow;
    private final CircularFifoQueue<float[]> smoothWindow;

    private final Session session;
    private final Shape inputShape;
    private final SavedModelBundle model;
    private final TensorInfo inputDetail;
    private final TensorInfo outputDetail;

    private int minKwMs = 100;
    private int blockMs = 20;
    private int maxKwCnt = 1;
    private int lookahead = 0;
    private int tailRoomMs = 100;
    private int lookaheadMs = 0;

    /**
     * Total of blocks in an utterance
     */
    private int utteranceBlocks = 0;

    private float scoreThreshold = 0.01f;
    private float hitThreshold = 7.0f;

    private boolean addSoftmax = false;
    private boolean silenceOff = true;
    private boolean immediateTrigger = true;

    private String scoreStrategy = "hit_ratio";


    /**
     * @param modelPath path of tensorflow model directory
     * @param labels    classification labels, example: [SILENCE, NOT_KW, 'keyword1', 'keyword2']
     */
    public TfKws(String modelPath, String[] labels) {
        this.labels = Arrays.asList(labels);
        this.notKwId = this.labels.indexOf(NOT_KW);
        this.silenceId = this.labels.indexOf(SILENCE);
        this.keywordsIdx = IntStream.range(0, this.labels.size())
                .filter(this::isKwId)
                .boxed()
                .toList()
                .stream()
                .mapToInt(Integer::intValue)
                .toArray();

        for (String keyword : this.labels.stream().filter(this::isKw).toList()) {
            this.utteranceScores.put(keyword, 0.0f);
            this.utteranceHits.put(keyword, 0);
        }

        this.tailThreshold = this.tailRoomMs / this.blockMs;
        this.countKeywords = this.keywordsIdx.length;
        int ringLen = 4000 / this.blockMs; // 4s of records

        // Initialize with SILENCE
        this.labelRing = new CircularFifoQueue<>(ringLen);
        for (int i = 0; i < ringLen; i++) {
            this.labelRing.add(silenceId);
        }

        // win_smooth = 30
        this.smoothWindow = new CircularFifoQueue<>(30);
        for (int i = 0; i < 30; i++) {
            this.smoothWindow.add(new float[this.countKeywords]);
        }

        // win_max = 100 (follow Google DNN paper)
        this.scoreWindow = new CircularFifoQueue<>(100);
        for (int i = 0; i < 100; i++) {
            this.scoreWindow.add(new float[this.countKeywords]);
        }

        // Load the TensorFlow graph from a .pb file
        this.model = SavedModelBundle.load(modelPath, "serve");
        SignatureDef signatureDef = this.model.metaGraphDef()
                .getSignatureDefOrThrow("serving_default");
        this.session = this.model.session();
        this.inputDetail = new ArrayList<>(signatureDef.getInputsMap().values()).get(0);
        this.outputDetail = new ArrayList<>(signatureDef.getOutputsMap().values()).get(0);
        this.blockSize = this.inputDetail.getTensorShape().getDim(1).getSize();
        this.inputShape = Shape.of(1, this.blockSize);
    }

    public void close() {
        this.model.close();
    }

    /**
     * Process an audio block
     *
     * @param pcm input audio block data
     * @return keyword string when hit or None when not hit
     */
    public String[] process(float[] pcm) {

        try (TFloat32 inputTensor = TFloat32.tensorOf(this.inputShape)) {
            for (int i = 0; i < this.blockSize; i++) {
                inputTensor.setFloat(pcm[i], 0, i);
            }

            try (Result result = this.session.runner()
                    .feed(this.inputDetail.getName(), inputTensor)
                    .fetch(this.outputDetail.getName())
                    .run()) {

                FloatNdArray scores = (TFloat32) result.get(0); // Shape (1, 4)

                if (this.addSoftmax) {
                    scores = softmax(scores);
                }

                return anyKwTriggered(scores);
            } catch (Exception e) {
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new String[0];
    }

    private String[] anyKwTriggered(FloatNdArray scores) {
        int labelIndex = argmax(scores);
        String label = this.labels.get(labelIndex);
        this.labelRing.add(labelIndex);
        boolean hitRatioStrategy = this.scoreStrategy.equals("hit_ratio");

        float score;
        if (hitRatioStrategy) {
            score = scores.getFloat(0, labelIndex);
        } else {
            score = confidenceScore(scores);
        }

        if (!metEnterCond()) {
            return new String[0];
        }

        // Below only run in utterance state
        this.utteranceBlocks++;

        // Label is kw, record score
        if (isKwId(labelIndex)) {
            int lastKwMs = getLastKwMs();

            // Calculate score
            if (hitRatioStrategy) {
                if (score > this.hitThreshold) {
                    this.utteranceHits.put(label, this.utteranceHits.get(label) + 1);
                }
                int nMaxKw = Math.max(this.minKwMs / this.blockMs, lastKwMs / this.blockMs);
                this.utteranceScores.put(label, (float) this.utteranceHits.get(label) / nMaxKw);
            } else {
                this.utteranceScores.put(label, score); // Update to latest confidence
            }

            float calcScore = this.utteranceScores.get(label);

            if (hitRatioStrategy) {
                this.logger.log(Level.TRACE, String.format("hitScore=%.2f\tscore=%.2f\tlabel=%s",
                        score, calcScore, label));
            } else {
                this.logger.log(Level.TRACE, String.format("score=%.2f\tlabel=%s", calcScore, label));
            }

            // Kw trigger before utterance end
            if (calcScore > this.scoreThreshold && lastKwMs > this.minKwMs && !this.alreadyTriggered.contains(label)) {
                this.logger.debug(String.format("Early trigger of kw: %s, dur: %s, score: %s",
                        label, lastKwMs, calcScore));
                this.alreadyTriggered.add(label);

                if (this.immediateTrigger && this.alreadyTriggered.size() == this.maxKwCnt) {
                    this.logger.debug(String.format("[!] Return keywords before utterance end: %s",
                            String.join(" ", this.alreadyTriggered)));
                    return this.alreadyTriggered.toArray(new String[0]);
                }
            }
        }

        // End of utterance
        if (metEndCond()) {
            String[] kw;

            if (this.alreadyTriggered.isEmpty()
                    || (this.immediateTrigger && this.alreadyTriggered.size() == this.maxKwCnt)) {
                kw = new String[0];
            } else {
                kw = this.alreadyTriggered.toArray(new String[0]);
            }

            int utterMs = (this.utteranceBlocks * this.blockMs) - this.tailRoomMs;
            this.logger.debug(String.format("End of utterance: %s, dur: %s, scores: %s",
                    Arrays.toString(kw), utterMs, this.utteranceScores));

            resetStates();

            if (kw.length > 0) {
                this.logger.info(String.format("[!] Return keywords after utterance end: \"%s\"",
                        Arrays.toString(kw)));
            }

            return kw;
        }

        return new String[0];
    }

    public FloatNdArray softmax(FloatNdArray scores) {
        FloatNdArray result = NdArrays.ofFloats(scores.shape());
        float sum = 0.0f;

        for (int i = 0; i < scores.shape().get(1); i++) {
            float exp = (float) Math.exp(scores.getFloat(0, i));
            result.setFloat(exp, 0, i);
            sum += exp;
        }

        for (int i = 0; i < scores.shape().get(1); i++) {
            result.setFloat(result.getFloat(0, i) / sum, 0, i);
        }

        return result;
    }

    private boolean isKw(String label) {
        return label != null && !label.equals(SILENCE) && !label.equals(NOT_KW);
    }

    private boolean isKwId(int labelId) {
        return labelId != silenceId && labelId != notKwId;
    }

    private int getLastKwMs() {
        Integer lastLabelId = this.labelRing.get(this.labelRing.size() - 1);

        if (lastLabelId == null) {
            return 0;
        }

        int cnt = 0;
        for (int i = labelRing.size() - 1; i >= 0; i--) {
            int labelId = labelRing.get(i);
            if (labelId == lastLabelId) {
                cnt++;
            } else {
                return cnt * this.blockMs;
            }
        }

        return 0;
    }

    private float confidenceScore(FloatNdArray scores) {
        // Discard non-kw scores
        float[] keywordsScores = new float[this.countKeywords];
        for (int i = 0; i < this.countKeywords; i++) {
            keywordsScores[i] = scores.getFloat(0, this.keywordsIdx[i]);
        }
        this.smoothWindow.add(keywordsScores);

        // Smoothed posterior of output probability at current frame
        float[] smoothedKeywordScores = new float[this.countKeywords];
        for (float[] kwScores : this.smoothWindow) {
            for (int i = 0; i < this.countKeywords; i++) {
                smoothedKeywordScores[i] += kwScores[i];
            }
        }
        for (int i = 0; i < this.countKeywords; i++) {
            smoothedKeywordScores[i] /= this.smoothWindow.size();
        }
        this.scoreWindow.add(smoothedKeywordScores);

        // Confidence score
        float confidence = 1.0f;
        for (int i = 0; i < this.countKeywords; i++) {
            float maxScore = Float.NEGATIVE_INFINITY;
            for (float[] smKwScores : this.scoreWindow) {
                if (smKwScores[i] > maxScore) {
                    maxScore = smKwScores[i];
                }
            }
            confidence *= maxScore;
        }

        return (confidence > 0) ? (float) Math.pow(confidence, 1.0 / this.countKeywords) : 0;
    }

    private void resetStates() {
        this.utteranceBlocks = 0;
        for (String k : this.utteranceScores.keySet()) {
            this.utteranceScores.put(k, 0.0f);
            this.utteranceHits.put(k, 0);
        }
        this.alreadyTriggered.clear();
        this.smoothWindow.clear();
        this.scoreWindow.clear();
        for (int i = 0; i < 30; i++) {
            this.smoothWindow.add(new float[this.countKeywords]);
        }
        for (int i = 0; i < 100; i++) {
            this.scoreWindow.add(new float[this.countKeywords]);
        }
    }

    /**
     * Met condition to enter/keep utterance state
     */
    private boolean metEnterCond() {
        if (this.utteranceBlocks == 0) { // not in utterance state, need to meet enter condition

            Integer lastLabelId = this.labelRing.get(this.labelRing.size() - 1);

            if (lastLabelId == null || !isKwId(lastLabelId)) {
                return false;
            }

            if (!this.silenceOff) {
                List<Integer> listLabelRing = new ArrayList<>(this.labelRing);
                List<Integer> head = listLabelRing.subList(
                        Math.max(0, listLabelRing.size() - 1 - this.lookahead), listLabelRing.size() - 1);
                // ignore kw prefixed by !kw if haven't entered utterance state
                long countSilence = head.stream().filter(v -> v == silenceId).count();
                return head.isEmpty() || !((double) countSilence / this.lookahead < 0.5);
            }
        }
        // can enter or just stays in utterance state
        return true;
    }

    private boolean metEndCond() {
        List<Integer> listLabelRing = new ArrayList<>(this.labelRing);
        Stream<Integer> stream = listLabelRing
                .subList(Math.max(0, listLabelRing.size() - this.tailThreshold), listLabelRing.size())
                .stream();

        // no kw after a certain time
        if (!this.silenceOff) {
            return stream.allMatch(v -> v == silenceId); // kw in sentence not accepted
        }

        return stream.allMatch(v -> v == silenceId || v == notKwId); // accept kw in sentence
    }

    public long getBlockSize() {
        return this.blockSize;
    }

    /**
     * @param scoreStrategy can be one of the following, 'smoothed_confidence': the score smoothing method used in
     *                      Google DDN paper, 'hit_ratio' (default): count frame scores over threshold
     */
    public void setScoreStrategy(String scoreStrategy) {

        if (!scoreStrategy.equals("hit_ratio") && !scoreStrategy.equals("smoothed_confidence")) {
            throw new RuntimeException("Unknown score strategy");
        }

        this.scoreStrategy = scoreStrategy;
    }

    /**
     * @param scoreThreshold threshold for kw hit ratio, or threshold for smoothed confidence (default 0.01)
     */
    public void setScoreThreshold(float scoreThreshold) {
        this.scoreThreshold = scoreThreshold;
    }

    /**
     * @param hitThreshold block score threshold to trigger kw hit, only used in hit_ratio, (default 7)
     */
    public void setHitThreshold(float hitThreshold) {
        this.hitThreshold = hitThreshold;
    }

    /**
     * @param tailRoomMs utterance end after how long of silence (default 100 ms)
     */
    public void setTailRoomMs(int tailRoomMs) {
        this.tailRoomMs = tailRoomMs;
        checkTailRoom();
    }

    /**
     * @param minKwMs minimum kw duration (default 100 ms)
     */
    public void setMinKwMs(int minKwMs) {
        this.minKwMs = minKwMs;
    }

    /**
     * @param blockMs block duration (default 20 ms), must match the model
     */
    public void setBlockMs(int blockMs) {
        this.blockMs = blockMs;
        checkTailRoom();
        calcLookahead();
    }

    /**
     * @param lookaheadMs silence lookahead window to prevent kw in sentence, 0 to turn off
     */
    public void setLookaheadMs(int lookaheadMs) {
        this.lookaheadMs = lookaheadMs;
        calcLookahead();
    }

    /**
     * @param maxKwCnt max keyword in one utterance
     */
    public void setMaxKwCnt(int maxKwCnt) {
        this.maxKwCnt = maxKwCnt;
    }

    /**
     * Disable trigger immediately once score reach threshold (don't wait for utterance end)
     */
    public void disableImmediateTrigger() {
        this.immediateTrigger = false;
    }

    /**
     * Add softmax layer to output
     */
    public void addSoftmaxLayerToOutput() {
        this.addSoftmax = true;
    }

    /**
     * Disable treat SILENCE as NOT_KW, turn silence detection off
     */
    public void disableSilenceOff() {
        this.silenceOff = false;
        if (!this.labels.contains(SILENCE)) {
            throw new RuntimeException("SILENCE must be in labels");
        }
    }

    private void checkTailRoom() {
        if (this.tailRoomMs <= 2 * this.blockMs) {
            throw new RuntimeException("TailRoom cannot be less than 2 x Block");
        }
    }

    private void calcLookahead() {
        this.lookahead = this.lookaheadMs / this.blockMs;
    }

    /**
     * Find the maximum probability and return its index.
     *
     * @param probabilities The probabilities.
     * @return The index of the max.
     */
    private int argmax(FloatNdArray probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.shape().get(1); i++) {
            float curVal = probabilities.getFloat(0, i);
            if (curVal > maxVal) {
                maxVal = curVal;
                idx = i;
            }
        }
        return idx;
    }
}
