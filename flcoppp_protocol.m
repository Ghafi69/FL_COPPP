function varargout = flcoppp_protocol(action,varargin)
%FLCOPPP_PROTOCOL Separated client/server integrity protocol for FL-COP++.
%
% This module deliberately separates four operations:
%   enroll    : client creates a Merkle-HMAC tree; server stores only root
%               and protocol metadata (no raw data and no tree nodes).
%   challenge : server issues a fresh nonce-bound challenge using a
%               cryptographic PRF sampler rather than MATLAB randperm.
%   prove     : client generates and AEAD-protects a proof packet.
%   verify    : server decrypts, parses, authenticates, checks replay state,
%               and verifies every Merkle path against the enrolled root.
%
% The protocol proves consistency/possession relative to the enrolled state.
% It does not prove that the committed bytes were used during local training.

action = lower(string(action));
switch action
    case "enroll"
        [varargout{1:nargout}] = enrollClient(varargin{:});
    case "challenge"
        [varargout{1:nargout}] = issueChallenge(varargin{:});
    case "prove"
        [varargout{1:nargout}] = clientGeneratePacket(varargin{:});
    case "verify"
        [varargout{1:nargout}] = serverVerifyPacket(varargin{:});
    case "selftest"
        [varargout{1:nargout}] = protocolSelfTest(varargin{:});
    case "exactdetection"
        varargout{1} = exactDetectionProbability(varargin{:});
    otherwise
        error('Unknown flcoppp_protocol action: %s',action);
end
end

%% Enrollment
function [clientState,serverRecord,registrationBytes] = enrollClient( ...
    committedBytes,sessionKey,blockSize,codeDigest,configDigest,clientID)

if nargin < 6, clientID = 0; end
committedBytes = uint8(committedBytes(:));
sessionKey = uint8(sessionKey(:));
codeDigest = uint8(codeDigest(:));
configDigest = uint8(configDigest(:));

if isempty(committedBytes)
    error('Cannot enroll an empty byte stream.');
end
if blockSize < 64 || blockSize ~= floor(blockSize)
    error('blockSize must be an integer >= 64 bytes.');
end

salt = sha256([uint8('FL-COP++ protocol-v5 salt')'; packU32(clientID)]);
leafKey = hkdfSHA256(sessionKey,salt,uint8('leaf-key-v5'),32);
tokenKey = hkdfSHA256(sessionKey,salt,uint8('proof-token-key-v5'),32);
challengeKey = hkdfSHA256(sessionKey,salt,uint8('challenge-prf-key-v5'),32);
channelKey = hkdfSHA256(sessionKey,salt,uint8('aead-channel-key-v5'),32);

blocks = splitBlocks(committedBytes,blockSize);
numBlocks = numel(blocks);
leaves = cell(numBlocks,1);
for i = 1:numBlocks
    tag = leafTag(leafKey,i,blocks{i});
    leaves{i} = sha256([uint8(0);tag(:)]);
end
levels = buildMerkleLevels(leaves);
root = uint8(levels{end}{1}(:));

clientState = struct();
clientState.ProtocolVersion = uint8(5);
clientState.ClientID = uint32(clientID);
clientState.TreeLevels = levels;
clientState.LeafKey = leafKey;
clientState.TokenKey = tokenKey;
clientState.ChannelKey = channelKey;
clientState.BlockSize = uint32(blockSize);
clientState.NumBlocks = uint32(numBlocks);
clientState.ByteLength = uint64(numel(committedBytes));
clientState.CodeDigest = uint8(codeDigest(:));
clientState.ConfigDigest = uint8(configDigest(:));

serverRecord = struct();
serverRecord.ProtocolVersion = uint8(5);
serverRecord.ClientID = uint32(clientID);
serverRecord.Root = root;
serverRecord.TokenKey = tokenKey;
serverRecord.ChallengeKey = challengeKey;
serverRecord.ChannelKey = channelKey;
serverRecord.BlockSize = uint32(blockSize);
serverRecord.NumBlocks = uint32(numBlocks);
serverRecord.ByteLength = uint64(numel(committedBytes));
serverRecord.CodeDigest = uint8(codeDigest(:));
serverRecord.ConfigDigest = uint8(configDigest(:));
serverRecord.IssuedCounter = uint64(0);
serverRecord.AcceptedCounter = uint64(0);
serverRecord.LastIssuedDigest = zeros(32,1,'uint8');

% Registration message: version, client id, root, sizes, code/config digests.
registration = [uint8(5);packU32(clientID);root(:);packU32(blockSize); ...
    packU32(numBlocks);packU64(numel(committedBytes));codeDigest(:);configDigest(:)];
serverRecord.RegistrationDigest = sha256(registration);
registrationBytes = numel(registration);
end

%% Server challenge
function [challenge,serverRecord,wireBytes] = issueChallenge( ...
    serverRecord,roundID,q)

N = double(serverRecord.NumBlocks);
q = min(max(1,round(double(q))),N);
serverRecord.IssuedCounter = serverRecord.IssuedCounter + uint64(1);
seq = serverRecord.IssuedCounter;

nonce = secureRandomBytes(16);
aeadNonce = secureRandomBytes(12);
indices = prfSampleWithoutReplacement(serverRecord.ChallengeKey,nonce, ...
    double(serverRecord.ClientID),double(roundID),seq,N,q);

challenge = struct();
challenge.ProtocolVersion = uint8(5);
challenge.ClientID = serverRecord.ClientID;
challenge.RoundID = uint32(roundID);
challenge.Sequence = seq;
challenge.Nonce = nonce(:);
challenge.AEADNonce = aeadNonce(:);
challenge.Indices = uint32(indices(:));
challenge.Serialized = serializeChallenge(challenge);
challenge.Digest = sha256(challenge.Serialized);

serverRecord.LastIssuedDigest = challenge.Digest;
wireBytes = numel(challenge.Serialized);
end

%% Client proof generation
function [packet,diag] = clientGeneratePacket(clientState,currentBytes,challenge,varargin)
currentBytes = uint8(currentBytes(:));
validateChallengeForClient(clientState,challenge);
if numel(varargin)>=1, currentCodeDigest=uint8(varargin{1}(:)); else, currentCodeDigest=clientState.CodeDigest; end
if numel(varargin)>=2, currentConfigDigest=uint8(varargin{2}(:)); else, currentConfigDigest=clientState.ConfigDigest; end
if numel(currentCodeDigest)~=32 || numel(currentConfigDigest)~=32
    error('Current code/config digests must each contain 32 bytes.');
end

q = numel(challenge.Indices);
body = [uint8(5);packU64(numel(currentBytes));packU16(q); ...
    currentCodeDigest(:);currentConfigDigest(:)];
pathHashes = 0;

for k = 1:q
    idx = double(challenge.Indices(k));
    if idx < 1 || idx > double(clientState.NumBlocks)
        error('Challenge index outside enrolled tree.');
    end
    block = getBlock(currentBytes,idx,double(clientState.BlockSize));
    tag = leafTag(clientState.LeafKey,idx,block);
    leaf = sha256([uint8(0);tag(:)]);
    [path,sides] = merklePath(clientState.TreeLevels,idx);

    body = [body;packU32(idx);leaf(:);packU16(numel(path))]; %#ok<AGROW>
    for p = 1:numel(path)
        body = [body;uint8(sides(p));uint8(path{p}(:))]; %#ok<AGROW>
    end
    pathHashes = pathHashes + numel(path);
end

token = hmacSHA256(clientState.TokenKey,[challenge.Digest(:);body(:)]);
plaintext = [body(:);token(:)];
aad = proofAAD(challenge,clientState.CodeDigest,clientState.ConfigDigest);
ciphertext = aeadEncrypt(plaintext,clientState.ChannelKey,challenge.AEADNonce,aad);

packet = struct();
packet.ProtocolVersion = uint8(5);
packet.Ciphertext = uint8(ciphertext(:));
packet.WireBytes = 1 + 4 + numel(packet.Ciphertext); % version + length + payload

diag = struct();
diag.CurrentByteLength = numel(currentBytes);
diag.Challenges = q;
diag.PathHashes = pathHashes;
diag.PlaintextBytes = numel(plaintext);
diag.CiphertextBytesIncludingTag = numel(ciphertext);
diag.WireBytes = packet.WireBytes;
end

%% Server verification
function [pass,serverRecord,diag] = serverVerifyPacket(serverRecord,challenge,packet)
pass = false;
diag = defaultVerifyDiag();

try
    if challenge.ProtocolVersion ~= serverRecord.ProtocolVersion || ...
            packet.ProtocolVersion ~= serverRecord.ProtocolVersion
        diag.FailureReason = "protocol_version"; return;
    end
    if challenge.ClientID ~= serverRecord.ClientID
        diag.FailureReason = "client_id"; return;
    end
    challengeBytes = serializeChallenge(challenge);
    challengeDigest = sha256(challengeBytes);
    if ~constantTimeEqual(challengeDigest,challenge.Digest) || ...
            ~constantTimeEqual(challengeDigest,serverRecord.LastIssuedDigest)
        diag.FailureReason = "challenge_digest"; return;
    end
    if challenge.Sequence ~= serverRecord.IssuedCounter || ...
            challenge.Sequence <= serverRecord.AcceptedCounter
        diag.ReplayRejected = true;
        diag.FailureReason = "replay_or_sequence"; return;
    end

    aad = proofAAD(challenge,serverRecord.CodeDigest,serverRecord.ConfigDigest);
    [plaintext,aeadOK] = aeadDecrypt(packet.Ciphertext,serverRecord.ChannelKey, ...
        challenge.AEADNonce,aad);
    diag.AEADValid = aeadOK;
    if ~aeadOK
        consumeSequence();
        diag.FailureReason = "aead"; return;
    end

    [parsed,parseOK] = parseProofPlaintext(plaintext);
    diag.ParseValid = parseOK;
    if ~parseOK
        consumeSequence();
        diag.FailureReason = "parse"; return;
    end

    expectedToken = hmacSHA256(serverRecord.TokenKey, ...
        [challenge.Digest(:);parsed.BodyBytes(:)]);
    diag.TokenValid = constantTimeEqual(parsed.Token,expectedToken);
    if ~diag.TokenValid
        consumeSequence();
        diag.FailureReason = "token"; return;
    end

    diag.LengthValid = parsed.CurrentByteLength == serverRecord.ByteLength;
    if ~diag.LengthValid
        consumeSequence();
        diag.FailureReason = "byte_length"; return;
    end
    diag.CodeDigestValid = constantTimeEqual(parsed.CodeDigest,serverRecord.CodeDigest);
    diag.ConfigDigestValid = constantTimeEqual(parsed.ConfigDigest,serverRecord.ConfigDigest);
    if ~diag.CodeDigestValid || ~diag.ConfigDigestValid
        consumeSequence();
        diag.FailureReason = "code_or_config_digest"; return;
    end

    expectedIndices = uint32(challenge.Indices(:));
    diag.IndicesValid = isequal(parsed.Indices(:),expectedIndices(:));
    if ~diag.IndicesValid
        consumeSequence();
        diag.FailureReason = "indices"; return;
    end

    pathsOK = true;
    for k = 1:numel(parsed.Indices)
        pathsOK = pathsOK && verifyMerklePath(parsed.Leaves{k}, ...
            parsed.Paths{k},parsed.Sides{k},serverRecord.Root);
    end
    diag.PathsValid = pathsOK;
    if ~pathsOK
        consumeSequence();
        diag.FailureReason = "merkle_path"; return;
    end

    pass = true;
    serverRecord.AcceptedCounter = challenge.Sequence;
    diag.FailureReason = "none";
catch ME
    consumeSequence();
    diag.FailureReason = "exception:" + string(ME.identifier);
end

    function consumeSequence()
        % A challenge is single-use even if verification fails. This prevents
        % repeated adaptive attempts with the same nonce and index set.
        serverRecord.AcceptedCounter = max(serverRecord.AcceptedCounter,challenge.Sequence);
    end
end

%% Self-test
function report = protocolSelfTest(sessionKey,blockSize,codeDigest,configDigest)
if nargin < 1 || isempty(sessionKey), sessionKey = secureRandomBytes(32); end
if nargin < 2 || isempty(blockSize), blockSize = 256; end
if nargin < 3, codeDigest = sha256(uint8('selftest-code')); end
if nargin < 4, configDigest = sha256(uint8('selftest-config')); end

bytes = uint8(mod((0:8191)',251));
[client,server] = enrollClient(bytes,sessionKey,blockSize,codeDigest,configDigest,7);
[ch,server] = issueChallenge(server,1,8);
[pkt,~] = clientGeneratePacket(client,bytes,ch);
[pass,server,diag] = serverVerifyPacket(server,ch,pkt);

% Replay must fail.
[replayPass,~,replayDiag] = serverVerifyPacket(server,ch,pkt);

% Ciphertext modification must fail AEAD authentication.
[ch2,server2] = issueChallenge(server,2,8);
[pkt2,~] = clientGeneratePacket(client,bytes,ch2);
mid=max(1,floor(numel(pkt2.Ciphertext)/2));
pkt2.Ciphertext(mid)=bitxor(pkt2.Ciphertext(mid),uint8(1));
[tamperPacketPass,server2,tamperDiag] = serverVerifyPacket(server2,ch2,pkt2); %#ok<ASGLU>

% Modify a definitely challenged block: Merkle verification must fail.
[ch3,server3] = issueChallenge(server2,3,8);
changed = bytes;
idx = double(ch3.Indices(1));
pos = min(numel(changed),(idx-1)*blockSize+1);
changed(pos) = bitxor(changed(pos),uint8(3));
[pkt3,~] = clientGeneratePacket(client,changed,ch3);
[changedPass,server3,changedDiag] = serverVerifyPacket(server3,ch3,pkt3);

% Re-encrypt a packet after changing only its token. AEAD must pass while
% the independent server-side token recomputation rejects it.
[ch4,server4] = issueChallenge(server3,4,8);
[pkt4,~] = clientGeneratePacket(client,bytes,ch4);
aad4=proofAAD(ch4,client.CodeDigest,client.ConfigDigest);
[plain4,ok4]=aeadDecrypt(pkt4.Ciphertext,client.ChannelKey,ch4.AEADNonce,aad4);
if ~ok4, error('Self-test could not decrypt its own packet.'); end
plain4(end)=bitxor(plain4(end),uint8(1));
pkt4.Ciphertext=aeadEncrypt(plain4,client.ChannelKey,ch4.AEADNonce,aad4);
pkt4.WireBytes=1+4+numel(pkt4.Ciphertext);
[tokenPass,server4,tokenDiag]=serverVerifyPacket(server4,ch4,pkt4);

% A changed submitted configuration digest must be rejected. This checks
% digest consistency only; it is not a proof that the code was executed.
[ch5,server5]=issueChallenge(server4,5,8);
badConfig=client.ConfigDigest; badConfig(1)=bitxor(badConfig(1),uint8(1));
[pkt5,~]=clientGeneratePacket(client,bytes,ch5,client.CodeDigest,badConfig);
[configPass,~,configDiag]=serverVerifyPacket(server5,ch5,pkt5);

report = struct();
report.ValidProofPass = pass;
report.ValidProofDiagnostic = diag;
report.ReplayRejected = ~replayPass && replayDiag.ReplayRejected;
report.CiphertextTamperRejected = ~tamperPacketPass && ~tamperDiag.AEADValid;
report.ChallengedBlockTamperRejected = ~changedPass && ~changedDiag.PathsValid;
report.TokenTamperRejected = ~tokenPass && tokenDiag.AEADValid && ~tokenDiag.TokenValid;
report.ConfigDigestTamperRejected = ~configPass && ~configDiag.ConfigDigestValid;
report.Passed = report.ValidProofPass && report.ReplayRejected && ...
    report.CiphertextTamperRejected && report.ChallengedBlockTamperRejected && ...
    report.TokenTamperRejected && report.ConfigDigestTamperRejected;
end

%% Serialization and parsing
function bytes = serializeChallenge(c)
bytes = [uint8(c.ProtocolVersion);packU32(c.ClientID);packU32(c.RoundID); ...
    packU64(c.Sequence);packU16(numel(c.Indices));uint8(c.Nonce(:)); ...
    uint8(c.AEADNonce(:))];
for i = 1:numel(c.Indices)
    bytes = [bytes;packU32(c.Indices(i))]; %#ok<AGROW>
end
end

function aad = proofAAD(challenge,codeDigest,configDigest)
aad = [uint8('FL-COP++-proof-v5')';challenge.Digest(:); ...
    uint8(codeDigest(:));uint8(configDigest(:))];
end

function [parsed,ok] = parseProofPlaintext(bytes)
parsed = struct(); ok = false;
bytes = uint8(bytes(:));
try
    if numel(bytes) < 1+8+2+32+32+32, return; end
    cursor = 1;
    version = bytes(cursor); cursor = cursor+1;
    if version ~= 5, return; end
    [byteLength,cursor] = readU64(bytes,cursor);
    [q,cursor] = readU16(bytes,cursor);
    q = double(q);
    requireBytes(64);
    codeDigest=bytes(cursor:cursor+31); cursor=cursor+32;
    configDigest=bytes(cursor:cursor+31); cursor=cursor+32;

    indices = zeros(q,1,'uint32');
    leaves = cell(q,1); paths = cell(q,1); sides = cell(q,1);
    for k = 1:q
        [idx,cursor] = readU32(bytes,cursor);
        indices(k) = idx;
        requireBytes(32);
        leaves{k} = bytes(cursor:cursor+31); cursor = cursor+32;
        [pathLen,cursor] = readU16(bytes,cursor);
        pathLen = double(pathLen);
        p = cell(pathLen,1); s = false(pathLen,1);
        for j = 1:pathLen
            requireBytes(33);
            s(j) = logical(bytes(cursor)); cursor = cursor+1;
            p{j} = bytes(cursor:cursor+31); cursor = cursor+32;
        end
        paths{k}=p; sides{k}=s;
    end

    bodyEnd = cursor-1;
    requireBytes(32);
    token = bytes(cursor:cursor+31); cursor = cursor+32;
    if cursor ~= numel(bytes)+1, return; end

    parsed.CurrentByteLength = byteLength;
    parsed.CodeDigest = codeDigest;
    parsed.ConfigDigest = configDigest;
    parsed.Indices = indices;
    parsed.Leaves = leaves;
    parsed.Paths = paths;
    parsed.Sides = sides;
    parsed.BodyBytes = bytes(1:bodyEnd);
    parsed.Token = token;
    ok = true;
catch
    ok = false;
end

    function requireBytes(n)
        if cursor+n-1 > numel(bytes)
            error('FLCOPPP:TruncatedProof','Truncated proof packet.');
        end
    end
end

%% Merkle helpers
function levels = buildMerkleLevels(leaves)
levels = cell(0,1);
current = leaves;
while numel(current)>1
    if mod(numel(current),2)==1
        current{end+1,1}=current{end};
    end
    levels{end+1,1}=current; %#ok<AGROW>
    parent=cell(numel(current)/2,1);
    for i=1:2:numel(current)
        parent{(i+1)/2}=sha256([uint8(1);current{i}(:);current{i+1}(:)]);
    end
    current=parent;
end
levels{end+1,1}=current;
end

function [path,sides] = merklePath(levels,index)
path=cell(numel(levels)-1,1);
sides=false(numel(levels)-1,1);
idx=index;
for l=1:numel(levels)-1
    nodes=levels{l};
    if mod(idx,2)==0
        sibling=idx-1; sides(l)=true;
    else
        sibling=idx+1; sides(l)=false;
    end
    sibling=min(sibling,numel(nodes));
    path{l}=nodes{sibling};
    idx=ceil(idx/2);
end
end

function ok = verifyMerklePath(leaf,path,sides,root)
node=uint8(leaf(:));
for i=1:numel(path)
    sibling=uint8(path{i}(:));
    if sides(i)
        node=sha256([uint8(1);sibling;node]);
    else
        node=sha256([uint8(1);node;sibling]);
    end
end
ok=constantTimeEqual(node,root);
end

function tag = leafTag(key,index,block)
tag=hmacSHA256(key,[uint8('leaf-v5')';packU32(index);uint8(block(:))]);
end

function blocks = splitBlocks(bytes,blockSize)
bytes=uint8(bytes(:));
N=max(1,ceil(numel(bytes)/blockSize));
blocks=cell(N,1);
for i=1:N
    blocks{i}=getBlock(bytes,i,blockSize);
end
end

function block = getBlock(bytes,index,blockSize)
bytes=uint8(bytes(:));
s=(index-1)*blockSize+1;
e=min(numel(bytes),index*blockSize);
if s>numel(bytes)
    block=uint8([]);
else
    block=bytes(s:e);
end
end

%% CSPRNG/PRF challenge sampling
function indices = prfSampleWithoutReplacement(key,nonce,clientID,roundID,seq,N,q)
selected=false(N,1); indices=zeros(q,1); count=0; counter=uint32(0);
base=[uint8('challenge-v5')';uint8(nonce(:));packU32(clientID); ...
    packU32(roundID);packU64(seq)];
while count<q
    counter=counter+uint32(1);
    digest=hmacSHA256(key,[base;packU32(counter)]);
    for off=1:4:29
        v=readU32At(digest,off);
        idx=mod(double(v),N)+1;
        if ~selected(idx)
            count=count+1; indices(count)=idx; selected(idx)=true;
            if count==q, break; end
        end
    end
end
end

function validateChallengeForClient(clientState,challenge)
if challenge.ProtocolVersion ~= clientState.ProtocolVersion
    error('Protocol version mismatch.');
end
if challenge.ClientID ~= clientState.ClientID
    error('Challenge addressed to a different client.');
end
expected=sha256(serializeChallenge(challenge));
if ~constantTimeEqual(expected,challenge.Digest)
    error('Challenge digest mismatch.');
end
end

%% AEAD and KDF helpers
function ciphertext = aeadEncrypt(plaintext,key,nonce,aad)
plaintext=uint8(plaintext(:)'); key=uint8(key(:)');
nonce=uint8(nonce(:)'); aad=uint8(aad(:)');
if numel(nonce)~=12, error('AES-GCM nonce must be 12 bytes.'); end
cipher=javax.crypto.Cipher.getInstance('AES/GCM/NoPadding');
spec=javax.crypto.spec.GCMParameterSpec(128,toJavaByteArray(nonce));
keySpec=javax.crypto.spec.SecretKeySpec(toJavaByteArray(key(1:32)),'AES');
cipher.init(javax.crypto.Cipher.ENCRYPT_MODE,keySpec,spec);
cipher.updateAAD(toJavaByteArray(aad));
ciphertext=javaBytesToUint8(cipher.doFinal(toJavaByteArray(plaintext)));
ciphertext=ciphertext(:);
end

function [plaintext,ok] = aeadDecrypt(ciphertext,key,nonce,aad)
plaintext=uint8([]); ok=false;
try
    cipher=javax.crypto.Cipher.getInstance('AES/GCM/NoPadding');
    spec=javax.crypto.spec.GCMParameterSpec(128,toJavaByteArray(uint8(nonce(:)')));
    keySpec=javax.crypto.spec.SecretKeySpec(toJavaByteArray(uint8(key(1:32)')),'AES');
    cipher.init(javax.crypto.Cipher.DECRYPT_MODE,keySpec,spec);
    cipher.updateAAD(toJavaByteArray(uint8(aad(:)')));
    plaintext=javaBytesToUint8(cipher.doFinal(toJavaByteArray(uint8(ciphertext(:)'))));
    plaintext=plaintext(:); ok=true;
catch
    plaintext=uint8([]); ok=false;
end
end

function okm = hkdfSHA256(ikm,salt,info,L)
prk=hmacSHA256(salt,ikm);
t=uint8([]); okm=uint8([]); counter=uint8(1);
while numel(okm)<L
    t=hmacSHA256(prk,[t(:);uint8(info(:));counter]);
    okm=[okm;t(:)]; %#ok<AGROW>
    counter=counter+uint8(1);
end
okm=okm(1:L);
end

function digest = sha256(bytes)
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(toJavaByteArray(uint8(bytes(:)')));
digest=javaBytesToUint8(md.digest()); digest=digest(:);
end

function tag = hmacSHA256(key,data)
mac=javax.crypto.Mac.getInstance('HmacSHA256');
ks=javax.crypto.spec.SecretKeySpec(toJavaByteArray(uint8(key(:)')),'HmacSHA256');
mac.init(ks);
tag=javaBytesToUint8(mac.doFinal(toJavaByteArray(uint8(data(:)')))); tag=tag(:);
end

function bytes = secureRandomBytes(n)
sr=java.security.SecureRandom();
b=zeros(1,n,'int8'); sr.nextBytes(b);
bytes=typecast(b,'uint8'); bytes=bytes(:);
end

function j = toJavaByteArray(bytes)
u=uint8(bytes(:)); j=reshape(typecast(u,'int8'),1,[]);
end

function u = javaBytesToUint8(jbytes)
j=int8(jbytes); u=typecast(j(:),'uint8'); u=u(:)';
end

function tf = constantTimeEqual(a,b)
a=uint8(a(:)); b=uint8(b(:));
if numel(a)~=numel(b), tf=false; return; end
acc=uint8(0);
for i=1:numel(a), acc=bitor(acc,bitxor(a(i),b(i))); end
tf=(acc==0);
end

%% Integer packing
function b=packU16(x)
x=uint16(x); b=uint8([bitshift(x,-8);bitand(x,uint16(255))]);
end
function b=packU32(x)
x=uint32(x); b=zeros(4,1,'uint8');
for i=1:4, b(i)=uint8(bitand(bitshift(x,-8*(4-i)),uint32(255))); end
end
function b=packU64(x)
x=uint64(x); b=zeros(8,1,'uint8');
for i=1:8, b(i)=uint8(bitand(bitshift(x,-8*(8-i)),uint64(255))); end
end
function [x,c]=readU16(b,c)
if c+1>numel(b), error('truncated'); end
x=bitor(bitshift(uint16(b(c)),8),uint16(b(c+1))); c=c+2;
end
function [x,c]=readU32(b,c)
if c+3>numel(b), error('truncated'); end
x=uint32(0);
for i=0:3, x=bitor(bitshift(x,8),uint32(b(c+i))); end
c=c+4;
end
function x=readU32At(b,c)
[x,~]=readU32(b,c);
end
function [x,c]=readU64(b,c)
if c+7>numel(b), error('truncated'); end
x=uint64(0);
for i=0:7, x=bitor(bitshift(x,8),uint64(b(c+i))); end
c=c+8;
end

%% Probability and diagnostics
function p = exactDetectionProbability(N,m,q)
N=double(N);m=double(m);q=min(double(q),N);
if m<=0, p=0; return; end
if q>N-m, p=1; return; end
logMiss=gammaln(N-m+1)-gammaln(q+1)-gammaln(N-m-q+1) ...
    - (gammaln(N+1)-gammaln(q+1)-gammaln(N-q+1));
p=min(max(1-exp(logMiss),0),1);
end

function d=defaultVerifyDiag()
d=struct('AEADValid',false,'ParseValid',false,'TokenValid',false, ...
    'LengthValid',false,'CodeDigestValid',false,'ConfigDigestValid',false, ...
    'IndicesValid',false,'PathsValid',false, ...
    'ReplayRejected',false,'FailureReason',"unverified");
end
