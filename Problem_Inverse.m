% ============================
% File: Problem_Inverse.m
% ============================
% RECONSTRUCTION EIT (problème inverse) — batch only
% -------------------------------------------------------------------------
% OBJECTIF
%   Lancer une reconstruction inverse EIT sur UNE slice CT en s'appuyant
%   sur les fichiers produits par Problem_Foward.m :
%     - Paramétrisation au choix : θ par TRIANGLE (recommandé) ou σ NODALE
%     - Pipeline simple : 0..N pas NOSER puis 0..M itérations GN
%     - Sauvegarde d'un pack unique + snapshots d'itération
%
% SORTIES (arborescence cible)
%   Outputs/<patient_this>/slice_<ZZZ>/inverse/prev_<patientPrev>_<PPP>/
%       ├─ reconstruction_inverse.mat   % struct 'rec' (résultats)
%       └─ plots_inverse/iterations/iter_###.mat  % snapshots nodal/tri par itération
%
% PRÉREQUIS
%   - Avoir déjà exécuté Problem_Foward.m pour la slice cible (pack forward
%     et mesh présents dans Outputs/<patient_this>/slice_<ZZZ>).
%   - OOEIT (ForwardMesh1st, EITFEM, …) accessible dans le path.
%   - src/ contient simulate_inverse_slice.m et ses dépendances.
%
% CONSEILS / PIÈGES
%   - INIT_MODE='morpho' nécessite un couple (patient_prev,slice_prev) valide
%     ayant un pack/mesh forward existant (même patient recommandé).
%   - BEST_MU et BEST_ALPHA sont des points de départ "raisonnables".
%     Ajuster si divergence/stagnation (voir misfit et snapshots).
%   - Aucune figure ne s'affiche ici (plots dans un autre script si besoin).
% -------------------------------------------------------------------------

clear; close all; clc;
addpath('src', genpath('src'));
addpath(genpath('/Users/anis/Documents/StageInria/Code/OOEIT-main'));  % adapter à votre installation

set(groot,'defaultFigureVisible','off');  % aucune figure ici

%% -------- Choix patients / slices ---------------------------------------
% Cible (à reconstruire)
patient_this = 's0011';
slice_this   = 301;

% Source pour l'initialisation morphologique (si INIT_MODE='morpho')
patient_prev = 's0011';
slice_prev   = 310;

%% ====== Hyperparamètres globaux =========================================
% Heuristiques "best" par défaut (à affiner selon vos données)
BEST_MU    = 0.15;     % μ* NOSER (sera re-scalé si 'scaled-median')
BEST_ALPHA = 1e-5;     % α* (ridge NOSER + régularisation GN)

% Planning d'itérations
N_NOSER = 1;           % # pas NOSER   (0 => pas de NOSER)
N_GN    = 0;           % # itérations GN (0 => pas de GN)

% Initialisation : 'morpho' (depuis slice_prev) ou 'constant'
INIT_MODE   = 'morpho';   % 'morpho' | 'constant'
SOFT_VALUE  = 0.35;       % σ homogène si INIT_MODE='constant' (S/m)

%% -------- Arborescence E/S ----------------------------------------------
rootThis = fullfile('Outputs',patient_this,sprintf('slice_%03d',slice_this));              % pack/mesh forward
invRoot  = fullfile(rootThis,'inverse',sprintf('prev_%s_%03d',patient_prev,slice_prev));   % sous-dossier dédié
plotDir  = fullfile(invRoot,'plots_inverse');
iterDir  = fullfile(plotDir,'iterations');

if ~exist(invRoot,'dir'), mkdir(invRoot); end
if ~exist(plotDir,'dir'), mkdir(plotDir); end
if ~exist(iterDir,'dir'), mkdir(iterDir); end

packFile = fullfile(rootThis,'eit_pack.mat');     % pack forward (indispensable)
recFile  = fullfile(invRoot,'reconstruction_inverse.mat');

assert(isfile(packFile), 'Pack forward introuvable: %s', packFile);
S = load(packFile);  % charge g,H,E,params, Vmat/Imeas/Ipat, sigma_tri, domain, etc.

%% ----- Clip dynamique depuis les conductivités (robuste) -----------------
% Borne des σ dans un intervalle un peu plus large que [min(cond), max(cond)]
vals = struct2array(S.params.cond);
lo = max(1e-3, min(vals)); 
hi = max(vals);
span = hi - lo;
lo = max(1e-3, lo - 0.02*span);
hi = hi + 0.02*span;
CLIP = [lo, hi];

fprintf('[INVERSE] target=(%s,%03d) | prev=(%s,%03d)\n', ...
        patient_this, slice_this, patient_prev, slice_prev);
fprintf('[INVERSE] Using fixed best: mu=%.6g, alpha=%.6g | clip=[%.4g, %.4g]\n', ...
        BEST_MU, BEST_ALPHA, CLIP(1), CLIP(2));
fprintf('[INVERSE] Plan: NOSER=%d step(s), GN=%d it(s)\n', N_NOSER, N_GN);
fprintf('[INVERSE] Output dir: %s\n', invRoot);

%% ====== Options reconstruction (passées à simulate_inverse_slice) =======
opts = struct();
opts.clip             = CLIP;     % bornes σ nodale
opts.doWarp           = true;     % warp contours (init morpho)
opts.cleanIterDir     = true;     % purge snapshots avant relance
opts.constNoise       = 3e-5;     % bruit additif (Gamma^-1)
opts.relNoise         = 1.2e-2;   % bruit relatif   (Gamma^-1)
opts.snapshots_enable = true;     % sauver iter_###.mat à chaque étape

% Paramétrisation / régularisation
opts.param_by    = 'tri';     % 'tri' (θ=σ par triangle, recommandé) | 'node' (σ nodale, debug)
opts.reg_on      = 'theta';   % 'theta' => α||θ||^2   |  'sigma' => α||σ||^2 (σ=Φθ)
opts.debug_fdJ   = false;     % true => check Jacobien chaîne Φ vs FD
opts.debug_fdJ_q = 8;         % nb colonnes FD si debug_fdJ

% ---- Initialisation
switch lower(INIT_MODE)
    case 'constant'
        % Init homogène : σ(x) = SOFT_VALUE
        opts.init = struct( ...
            'constant_enable', true, ...
            'soft_value',      SOFT_VALUE, ...
            'discrete',        true, ...          % sans effet ici
            'knn_k',           3, ...
            'quant',           struct('clean_iters',2,'centers',[]), ...
            'auto_blend',      false, ...
            'blend_max',       0, ...
            'blend_homog',     0, ...
            'smooth_iters',    0, ...
            'smooth_lambda',   0, ...
            'calib_enable',    false, ...
            'calib_clip',      [1 1], ...
            'noser_warmstart', struct('enable',false) );
    otherwise
        % Init morpho discrète depuis slice_prev (k-NN + majorité, etc.)
        opts.init = struct( ...
            'constant_enable', false, ...
            'discrete',true,'knn_k',3, ...
            'quant',struct('clean_iters',2,'centers',[]), ...
            'auto_blend',false,'blend_max',0,'blend_homog',0, ...
            'smooth_iters',0,'smooth_lambda',0, ...
            'calib_enable',false,'calib_clip',[1 1], ...
            'noser_warmstart',struct('enable',false) );
end

% ---- NOSER multi-steps
opts.noser_only = struct( ...
    'enable',       N_NOSER>0, ...
    'iters',        max(0,N_NOSER), ...
    'lambda',       BEST_MU, ...
    'lambda_mode',  'scaled-median', ...   % robuste à l'échelle du problème
    'use_diag',     true, ...
    'max_rel_step', 0.20, ...
    'linesearch',   false, ...
    'alpha_l2',     BEST_ALPHA);           % ridge léger au NOSER

% ---- GN (sur θ, avec éventuelle régularisation L2)
opts.iters_per_stage = max(0,N_GN);
opts.reg_alpha       = (N_GN>0) * BEST_ALPHA;

%% ====== Lancement du pipeline (NOSER puis GN) ===========================
% simulate_inverse_slice orchestre tout (lecture pack/mesh, init, Φ, FEM, etc.)
rec = simulate_inverse_slice(patient_this, slice_this, patient_prev, slice_prev, opts, iterDir);

%% ====== Sauvegarde finale ===============================================
save(recFile, '-struct', 'rec');

% Logging simple
mis = NaN;
if isfield(rec,'misfit') && ~isempty(rec.misfit), mis = rec.misfit; end
fprintf('\n[INVERSE] Reco terminée. Misfit relatif = %.6g\nFichier: %s\n', mis, recFile);
