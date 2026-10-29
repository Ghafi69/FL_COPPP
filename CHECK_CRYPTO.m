%% CHECK_CRYPTO.m
% Standalone Java cryptography check used by FL-COP++.
clear; clc;

assert(usejava('jvm')==1,'MATLAB JVM is required.');

fprintf('Java version: %s\n',char(java.lang.System.getProperty('java.version')));

rng(17,'twister');
for trial = 1:10
    payload = uint8(randi([0 255],1,trial*113));
    key = uint8(randi([0 255],1,32));
    nonce = uint8(randi([0 255],1,12));
    aad = uint8(randi([0 255],1,64));

    javaKey = toJavaByteArrayLocal(key);
    javaNonce = toJavaByteArrayLocal(nonce);
    javaAAD = toJavaByteArrayLocal(aad);
    javaPayload = toJavaByteArrayLocal(payload);

    enc = javax.crypto.Cipher.getInstance('AES/GCM/NoPadding');
    spec = javax.crypto.spec.GCMParameterSpec(128,javaNonce);
    keySpec = javax.crypto.spec.SecretKeySpec(javaKey,'AES');
    enc.init(javax.crypto.Cipher.ENCRYPT_MODE,keySpec,spec);
    enc.updateAAD(javaAAD);
    ciphertext = enc.doFinal(javaPayload);

    dec = javax.crypto.Cipher.getInstance('AES/GCM/NoPadding');
    spec2 = javax.crypto.spec.GCMParameterSpec(128,javaNonce);
    dec.init(javax.crypto.Cipher.DECRYPT_MODE,keySpec,spec2);
    dec.updateAAD(javaAAD);
    recovered = javaBytesToUint8Local(dec.doFinal(ciphertext));

    assert(isequal(uint8(recovered(:)'),uint8(payload(:)')), ...
        'AES-GCM mismatch at trial %d.',trial);
end

fprintf('AES-GCM: PASS (10/10 payloads)\n');

% SHA-256 consistency check.
message = uint8(0:255);
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(toJavaByteArrayLocal(message));
digest = javaBytesToUint8Local(md.digest());
assert(numel(digest)==32,'SHA-256 output length is not 32 bytes.');
fprintf('SHA-256: PASS\n');

% HMAC-SHA-256 consistency check.
mac = javax.crypto.Mac.getInstance('HmacSHA256');
ks = javax.crypto.spec.SecretKeySpec(toJavaByteArrayLocal(key),'HmacSHA256');
mac.init(ks);
tag = javaBytesToUint8Local(mac.doFinal(toJavaByteArrayLocal(message)));
assert(numel(tag)==32,'HMAC-SHA-256 output length is not 32 bytes.');
fprintf('HMAC-SHA-256: PASS\n\n');
fprintf('Cryptography check completed successfully.\n');

function j = toJavaByteArrayLocal(bytes)
u = uint8(bytes(:));
j = reshape(typecast(u,'int8'),1,[]);
end

function u = javaBytesToUint8Local(jbytes)
j = int8(jbytes);
u = typecast(j(:),'uint8');
u = u(:)';
end
