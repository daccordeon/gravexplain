close all
clc
clear all

% read sample files from optical microphone
aa_track = readtable('aa_melatos.csv');
aa = table2array(aa_track);
% choose the the first second to analyze (sampling frequency 16kHz)

fs = 16000;  % sampling frequency of photodiode
aa = aa(1:fs,2);
aa_inv =aa;
Fs = 44100;  % default sampling frequency of Matlab
% original voice sample (first second)
[melatos_source,FS] = audioread('source_melatos.wav',[1,Fs]);
melatos_source = melatos_source(:,1);
%melatos_source = melatos_source(1:length(aa),1);   % keep one voice chanel and "a cathode"
%audiowrite('melatos_source2.wav',melatos_source,fs)
%[melatos_source2,FS2] = audioread('melatos_source2.wav');

% pure system noise
podo = readtable('podo_14_6.csv');
podo1 = table2array(podo(1:fs,2));
%%
N = 15; % number of cascade notch filter
% spectrum of original human voice
T = 1/(FS); % sampling period
L = length(melatos_source);
figure(1)
subplot(7,2,1)
t = (0:L-1)*T;
% source with time shift
melatos_source_shift = [melatos_source(end-0.12*FS+1:end);melatos_source(1:end-0.12*FS)];
plot(t,melatos_source_shift)
title('Original voice')
xlabel('time(s)')
Y = fft(melatos_source,2^nextpow2(L));
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = FS*(0:(L/2))/L;

subplot(7,2,2)
plot(f,P1,'Linewidth',2) 
title('Original voice (frequency domain)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0,2500])
%% spectrum of recorded voice (with sampling frequency 16KHz)

T = 1/(fs); % sampling period
L = length(aa);
t = (0:L-1)*T;
Y_voice = fft(aa,2^nextpow2(length(aa)));
P2_voice = abs(Y_voice/L);
P1_voice = P2_voice(1:L/2+1);
P1_voice(2:end-1) = 2*P1_voice(2:end-1);
f = fs*(0:(L/2))/L;
subplot(7,2,4)
plot(f,P1_voice,'Linewidth',2) 
title('Raw output of optical microphone (frequency domain)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0,2500])
ylim([0,4])

%% plot of noise spectrum
L = length(podo1);
t = (0:L-1)*T;
Y_noise = fft(podo1,2^nextpow2(length(podo1)));
P2_noise = abs(Y_noise/L);
P1_noise = P2_noise(1:L/2+1);
P1_noise(2:end-1) = 2*P1_noise(2:end-1);
f = fs*(0:(L/2))/L;
subplot(7,2,6)
plot(f,P1_noise,'Linewidth',2) 
title('Background noise (frequency domain)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0,2500])
ylim([0,4])
kk = 2^nextpow2(length(podo1));
datanoise = kk*ifft(P1_voice,kk);
subplot(7,2,5)
plot(1/fs:1/fs:(kk-1)/fs,abs(datanoise(2:end)),'Linewidth',2)
xlabel('time(s)')
title('Background noise')
xlim([0,1])
ylim([0,400])
%% a naive idea is to substract noise spectrum directly
subplot(7,2,8)
plot(f,P1_voice-P1_noise,'Linewidth',2)
title('Background subtracted output (frequency domain)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0,2500])
ylim([0,4])
%%
k = 2^nextpow2(length(podo1));
data = k*ifft(P2_voice-P2_noise,k);
subplot(7,2,7)
plot(1/fs:1/fs:(k-1)/fs,abs(data(2:end)),'Linewidth',2)
xlabel('time(s)')
xlim([0,1])
ylim([0,300])
title('Background subtracted output')

%% recover filtered signal to time domain
% figure(2)
% subplot(211)
% plot(0:1/FS:1-1/FS,melatos_source)
% subplot(212)
% Y_cover = ifft(P1_voice-P1_noise,2^nextpow2(length(podo1)));
% plot(0:1/fs:length(Y_cover)/fs-1/fs,Y_cover)
%% find dominat noisy frequency bins
% find the maximum 10 frequency bins
% P1_sort = sort(P1,'descend');
% freq = [];
% P1_max10 = P1_sort(1:50);
% for x = 1: length(P1_max10)
%     freq = [freq,f(find(P1==P1_max10(x)))];
% end
% subplot(3,1,3)
% plot(freq,P1_max10,'o')
% xlim([0,1500])
% ylim([0,5])
% title('maximum frequency bins')



%% cascaded notch filter (N=5)
subplot(7,2,1)
plot(0:1/Fs:1-1/Fs,melatos_source,'Linewidth',2)
xlabel('time(s)')
title('Orginal voice')
subplot(7,2,3)
plot(0:1/fs:1-1/fs,aa,'Linewidth',2)
xlabel('time(s)')
title('Raw output of optical microphone')
%aa2 = aa;
for k=1:N
d(k) = designfilt('bandstopiir','FilterOrder',6,...
    'HalfPowerFrequency1',46*k,'HalfPowerFrequency2',54*k,...
    'DesignMethod','butter','SampleRate',fs);

aa = filtfilt(d(k),aa);
end
subplot(7,2,9)
plot(0:1/fs:1-1/fs,aa,'Linewidth',2)
title('Voice after notch filtering')
xlabel('time(s)')

%% Wiener filter
N=300; % Filter order
[xest,b,MSE] = wienerFilt(aa,aa-podo1,N);  % notch filter and Wiener filter combined 
[xest2,b2,MSE2] = wienerFilt(aa_inv,aa_inv-podo1,N);  % Only Wiener Filter
subplot(7,2,11)
%plot(0:1/fs:1-1/fs,aa)
%hold on
plot(0:1/fs:(16000-N-1)/fs,xest2)
title('Voice after Wiener filter')
xlabel('time(s)')
subplot(7,2,13)
plot(0:1/fs:(16000-N-1)/fs,xest)
title('Voice-combined Notch filter + Wiener filter')
xlabel('time(s)')
%legend('notch filter','Wiener filter','notch filter-Wiener filter')

%% plot frequency of filtered voice
T = 1/fs; % sampling period
clear f; clear P1
L = length(aa);
t = (0:L-1)*T;
Y = fft(aa-mean(aa));
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs*(0:(L/2))/L;
subplot(7,2,10)
plot(f,P1,'Linewidth',2) 
title(strcat('Voice after notch filter (frequency domain)'))
csvwrite('Voice after notch filter (frequency domain) (yaxis).txt',P1)
csvwrite('Voice after notch filter (frequency domain) (xaxis).txt',f)
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0,2500])
ylim([0,4])

%% ferquency domain of Wiener filetr
subplot(7,2,12)
Y_w2 = fft(xest2-mean(xest2));
P2_w2 = abs(Y_w2/44100);
P1_w2 = P2_w2(1:L/2+1);
P1_w2(2:end-1) = 2*P1_w2(2:end-1);
f = fs*(0:(L/2))/L;
plot(f,P1_w2,'Linewidth',2) 
xlim([0,2500])
ylim([0,4])
xlabel('f (Hz)')
ylabel('|P1(f)|')
title('Voice after Wiener filter (frequency domain)')

subplot(7,2,14)
T = 1/fs; % sampling period
L = length(xest);
t = (0:L-1)*T;
Y_w = fft(xest-mean(xest));
P2_w = abs(Y_w/44100);
P1_w = P2_w(1:L/2+1);
P1_w(2:end-1) = 2*P1_w(2:end-1);
f = fs*(0:(L/2))/L;
plot(f,P1_w,'Linewidth',2) 
title('Voice-combined notch + Wiener (frequency domain)')
xlim([0,2500])
ylim([0,4])
xlabel('f (Hz)')
ylabel('|P1(f)|')

%% plot notch filter response
% for k=1:5
% filter = designfilt('bandstopiir','FilterOrder',6,...
%     'HalfPowerFrequency1',46*k,'HalfPowerFrequency2',54*k,...
%     'DesignMethod','butter','SampleRate',fs);
% [hh,ww] = freqz(filter,10000);
% csvwrite(['notch filter ',num2str(k), '(x axis).txt'],ww/pi)
% csvwrite(['notch filter ',num2str(k), '(y axis).txt'],db(hh))
% subplot(5,1,k)
% plot(ww/pi,db(hh))
% axis([0 0.1 -50 10])
% xlabel('Normalized Frequency (\times\pi rad/sample)')
% ylabel('Magnitude (dB)')
% end