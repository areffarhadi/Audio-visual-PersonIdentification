function features = extractAuditorySpectrogram(x,fs)
%extractAuditorySpectrogram Compute auditory spectrogram
%
% features = extractAuditorySpectrogram(x,fs) computes an auditory (Bark)
% spectrogram in the same way as done in the Train Speech Command
% Recognition Model Using Deep Learning example. Specify the audio input,
% x, as a mono audio signal with a 1 second duration.

% Design audioFeatureExtractor object
persistent afe segmentSamples
if isempty(afe)
    designFs = 16e3;
    segmentDuration = 1;
    frameDuration = 0.025;
    hopDuration = 0.01;

    numBands = 50;
    FFTLength = 512;

    segmentSamples = round(segmentDuration*designFs);
    frameSamples = round(frameDuration*designFs);
    hopSamples = round(hopDuration*designFs);
    overlapSamples = frameSamples - hopSamples;

    afe = audioFeatureExtractor( ...
        SampleRate=designFs, ...
        FFTLength=FFTLength, ...
        Window=hann(frameSamples,"periodic"), ...
        OverlapLength=overlapSamples, ...
        barkSpectrum=true);
    setExtractorParams(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);
end

% Resample to 16 kHz if necessary
if double(fs)~=16e3
    x = cast(resample(double(x),16e3,double(fs)),like=x);
end

% Ensure the input is equal to 1 second of data at 16 kHz.
x = trimOrPad(x,segmentSamples);

% Extract features
features = extract(afe,x);

% Apply logarithm
features = log10(features + 1e-6);

end