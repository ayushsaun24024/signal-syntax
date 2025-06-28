export const cvContentForBio = `Professional Summary: Engineer & Applied Researcher with 2+ years turning data into production value through cloud-native ETL pipelines, big-data speech processing and robust data-modeling. Shipped 40+ zero-downtime releases at Samsung—launching a green-field portal in < 6 months and sustaining 99.8% uptime—before halving audio-deepfake error to EER 6.3% on 575 h of speech at IIIT-Delhi with reproducible PyTorch MLOps that ran experiments 3× faster. Owns the full lifecycle—data ingestion, feature engineering, model training, CI/CD, and observability—using Python, SQL, AWS, Docker, and Git to deliver measurable, production-ready impact.
Skills: Python, SQL, C++, Java, JavaScript, TypeScript, PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers, ETL Pipelines, Data Modeling, Hadoop, SpeechBrain, torchaudio, OpenCV, librosa, Pandas, NumPy, Matplotlib, CUDA, AWS (S3, EC2, IAM, CloudFront, CloudWatch), Docker, Jenkins, Git/GitHub, Linux, RESTful APIs, CI/CD, Jira, Confluence, React, Angular, Material-UI, Java Spring Boot, Swagger/OpenAPI, Jest, JUnit, Full-Stack Development.
Projects: Audio Deepfake Detection (Evaluated SOTA anti-spoofing models, Adapted SSL encoders with CNN heads, halving spoof EER to 6.3%), Automatic Speaker Verification System (Built end-to-end biometric speaker-verification pipeline, trimming tandem EER to 30%), Classic ML-Based Vocoder (Engineered a Mel-spectrogram to waveform vocoder with MOS 4.1), Single-Object Tracking System (Delivered a real-time tracker with camera-motion compensation sustaining 30 FPS).`;

export const skills = {
  'Programming Languages & Scripting': ['Python', 'SQL', 'C++', 'Java', 'JavaScript', 'TypeScript'],
  'AI, Machine Learning & Data Science': ['PyTorch', 'TensorFlow', 'scikit-learn', 'Hugging Face Transformers', 'ETL Pipelines', 'Data Modeling', 'Hadoop', 'SpeechBrain', 'torchaudio', 'OpenCV', 'librosa', 'Pandas', 'NumPy', 'Matplotlib', 'CUDA'],
  'Cloud & DevOps': ['AWS', 'Docker', 'Jenkins', 'Git/GitHub', 'Linux', 'RESTful APIs', 'CI/CD', 'Jira', 'Confluence'],
  'Web & Software Development': ['React.js', 'Angular.js', 'Material-UI', 'Java Spring Boot', 'Swagger/OpenAPI', 'Jest', 'JUnit', 'Full-Stack Development'],
  'Professional Skills': ['Research & Development', 'Agile/Scrum', 'Technical Documentation', 'Problem-Solving', 'Project Management', 'Cross-Functional Collaboration', 'Release Coordination', 'Presentation Skills'],
};

export const timeline = [
  {
    type: 'work',
    title: 'Post-Graduate Researcher',
    company: 'Infosys Centre for Artificial Intelligence, IIIT-Delhi',
    date: 'Jan 2025 - Present',
    description: 'Designed and implemented modular pipelines for speaker verification and anti-spoofing, significantly improving performance and efficiency. Fine-tuned SOTA models and applied advanced data augmentation techniques, achieving a 50% reduction in error rates.',
  },
  {
    type: 'education',
    title: 'M.Tech Computer Science',
    company: 'IIIT-Delhi',
    date: 'Aug 2024 - Present',
    description: 'Specializing in advanced computer science topics with a focus on artificial intelligence and machine learning.',
  },
  {
    type: 'work',
    title: 'Software Engineer',
    company: 'Samsung R&D Institute India - Delhi',
    date: 'Jun 2022 - Jul 2024',
    description: 'Architected and maintained internal portals, shipping 40+ releases with 99.8% uptime. Engineered responsive UIs with React.js, developed secure Spring Boot micro-services, and optimized databases, reducing latency by 40%.',
  },
  {
    type: 'education',
    title: 'B.Tech Electrical Engineering',
    company: 'Delhi Technological University',
    date: 'Aug 2018 - Jun 2022',
    description: 'Gained a strong foundation in engineering principles, with projects in software and hardware.',
  },
];

export type Project = {
  title: string;
  description: string;
  technologies: string[];
  link?: string;
  image: string;
  aiHint: string;
  details: {
    flowchart: string;
    pseudocode: string;
    insights: string;
  };
};

export const projects: Project[] = [
  {
    title: 'Audio Deepfake Detection',
    description: 'Evaluated and adapted SOTA anti-spoofing models and SSL encoders on 575h of speech, halving spoof EER to 6.3%. Optimized training using Bayesian search and integrated leading ML frameworks.',
    technologies: ['Python', 'PyTorch', 'SpeechBrain', 'HuggingFace Transformers', 'WavLM', 'ECAPA-TDNN', 'RawNet2', 'CUDA'],
    // link: 'https://github.com/ayushsaun24024',
    image: '/data/deepfake.png',
    aiHint: 'audio waveform',
    details: {
      flowchart: `graph TD
    A[Start: Audio Spoof Detection System] --> B{Initialize Global Config}
    B --> C[Load Foundation Model<br/>WavLM/Wav2Vec2]
    C --> D{Check Mode}
    
    D -->|TRAIN| E[Training Pipeline]
    D -->|TEST| F[Testing Pipeline]
    
    %% Training Pipeline
    E --> G[Load Training & Validation Data]
    G --> H[Initialize Nes2Net Model]
    H --> I[Initialize Classifier]
    I --> J[Initialize Optimizer & Loss Function]
    J --> K[Set best_accuracy = 0]
    K --> L[Start Epoch Loop]
    
    L --> M{epoch <= max_epochs?}
    M -->|Yes| N[Set total_loss = 0]
    M -->|No| Z1[Training Complete<br/>Return Models]
    
    N --> O[Start Batch Loop]
    O --> P{More batches?}
    P -->|Yes| Q[Audio Preprocessing & Augmentation]
    P -->|No| V1[Validation Check]
    
    %% Audio Preprocessing & Augmentation
    Q --> Q1[Load Raw Audio<br/>sample_rate=16000]
    Q1 --> Q2[Normalize Amplitude]
    Q2 --> Q3[Trim Silence]
    Q3 --> Q4{Length > max_length?}
    Q4 -->|Yes| Q5[Random Crop]
    Q4 -->|No| Q6[Pad with Zeros]
    Q5 --> Q7[Apply Augmentations]
    Q6 --> Q7
    
    Q7 --> Q8{Apply RawBoost?<br/>prob=0.7}
    Q8 -->|Yes| Q9[Select RawBoost Type<br/>1-7: Filter/Noise/Reverb/<br/>Scale/Echo/Clip/Compress]
    Q8 -->|No| Q10{Apply MUSAN?}
    Q9 --> Q10
    Q10 -->|Yes| Q11[Add MUSAN Noise<br/>Music/Speech/Noise<br/>SNR: 5-20dB]
    Q10 -->|No| Q12{Speed Perturbation?}
    Q11 --> Q12
    Q12 -->|Yes| Q13[Change Speed<br/>Factor: 0.9-1.1]
    Q12 -->|No| R[Feature Extraction]
    Q13 --> R
    
    %% Feature Extraction
    R --> R1[Convert to Tensor]
    R1 --> R2[Foundation Model Forward<br/>Extract SSL Features]
    R2 --> R3[Features Shape:<br/>batch x time x feature_dim]
    R3 --> S[Nes2Net Processing]
    
    %% Nes2Net Processing
    S --> S1[Transpose Features<br/>batch x feature_dim x time]
    S1 --> S2[Initialize nested_outputs list]
    S2 --> S3[Start Group Loop<br/>group_idx = 0]
    
    S3 --> S4{group_idx < num_groups?}
    S4 -->|Yes| S5[Nested Res2Net Block]
    S4 -->|No| S20[Global Average Pooling]
    
    S5 --> S6[Set residual = features]
    S6 --> S7[Calculate width = feature_dim/scale_factor]
    S7 --> S8[Split features by width]
    S8 --> S9[Start Scale Loop<br/>scale_idx = 0]
    
    S9 --> S10{scale_idx < scale_factor?}
    S10 -->|Yes| S11{scale_idx == 0?}
    S10 -->|No| S15[Concatenate Scale Outputs]
    
    S11 -->|Yes| S12[scale_out = feature_splits index 0]
    S11 -->|No| S13{scale_idx == 1?}
    S12 --> S14[Append to scale_outputs]
    S13 -->|Yes| S12A[scale_out = ReLU Conv1D feature_splits index 1]
    S13 -->|No| S12B[Nested Connection:<br/>input = feature_splits i + scale_outputs i-2<br/>scale_out = ReLU Conv1D input]
    S12A --> S14
    S12B --> S12C{scale_idx < scale_factor-1?}
    S12C -->|Yes| S12D[Additional Nested Conv<br/>scale_out = Conv1D scale_out + scale_out]
    S12C -->|No| S14
    S12D --> S14
    S14 --> S10
    
    S15 --> S16[Batch Normalization]
    S16 --> S17[Add Residual Connection<br/>nested_out = normalized + residual]
    S17 --> S18[Group Convolution]
    S18 --> S19{group_idx > 0?}
    S19 -->|Yes| S19A[Combine with Previous<br/>combined = concat nested_out + nested_outputs group_idx-1]
    S19 -->|No| S19B[nested_outputs append nested_out]
    S19A --> S19C[Fusion Layer<br/>fused = Conv1D combined]
    S19C --> S19D[nested_outputs append fused]
    S19D --> S19E[features = nested_outputs group_idx]
    S19B --> S19E
    S19E --> S19F[group_idx++]
    S19F --> S4
    
    S20 --> T[Classification]
    
    %% Classification
    T --> T1[Hidden Layer 1<br/>ReLU Linear pooled to 256]
    T1 --> T2[Dropout 0.3]
    T2 --> T3[Hidden Layer 2<br/>ReLU Linear hidden1 to 64]
    T3 --> T4[Dropout 0.3]
    T4 --> T5[Output Layer<br/>Linear hidden2 to 2]
    T5 --> U[Loss & Backpropagation]
    
    %% Loss & Backpropagation
    U --> U1[Calculate CrossEntropy Loss]
    U1 --> U2[total_loss += loss]
    U2 --> U3[optimizer zero_grad]
    U3 --> U4[loss backward]
    U4 --> U5[optimizer step]
    U5 --> P
    
    %% Validation
    V1 --> V2{epoch % validation_freq == 0?}
    V2 -->|Yes| V3[Set Model to Eval Mode]
    V2 -->|No| V20[Next Epoch]
    V3 --> V4[Initialize correct_predictions = 0<br/>total_samples = 0]
    V4 --> V5[Start Validation Batch Loop]
    
    V5 --> V6{More val_batches?}
    V6 -->|Yes| V7[Preprocess Audio<br/>NO AUGMENTATION]
    V6 -->|No| V15[Calculate Validation Accuracy]
    
    V7 --> V8[Center Crop/Pad Audio]
    V8 --> V9[Extract SSL Features]
    V9 --> V10[Nes2Net Processing<br/>Same as Training]
    V10 --> V11[Classification]
    V11 --> V12[Get Predictions<br/>argmax logits]
    V12 --> V13[Update Counters<br/>correct_predictions += correct<br/>total_samples += batch_size]
    V13 --> V6
    
    V15 --> V16[accuracy = correct/total * 100]
    V16 --> V17{accuracy > best_accuracy?}
    V17 -->|Yes| V18[Save Best Model<br/>Update best_accuracy]
    V17 -->|No| V19[Print Results]
    V18 --> V19
    V19 --> V20
    V20 --> L
    
    %% Testing Pipeline
    F --> F1[Load Trained Models<br/>best_nes2net.pth<br/>best_classifier.pth]
    F1 --> F2[Set Models to Eval Mode]
    F2 --> F3[Initialize test_results list]
    F3 --> F4[Start Test File Loop]
    
    F4 --> F5{More test files?}
    F5 -->|Yes| F6[Audio Preprocessing<br/>NO AUGMENTATION]
    F5 -->|No| F25[Return Test Results]
    
    F6 --> F7[Load & Normalize Audio]
    F7 --> F8[Trim Silence]
    F8 --> F9{Length > max_length?}
    F9 -->|Yes| F10[Center Crop]
    F9 -->|No| F11[Pad with Zeros]
    F10 --> F12[Extract SSL Features]
    F11 --> F12
    
    F12 --> F13[Nes2Net Processing<br/>Same Architecture as Training]
    F13 --> F14[Classification Forward Pass]
    F14 --> F15[Apply Softmax<br/>Get Probabilities]
    F15 --> F16[predicted_class = argmax probabilities]
    F16 --> F17[confidence = max probabilities]
    F17 --> F18{predicted_class == 0?}
    F18 -->|Yes| F19[result = bonafide]
    F18 -->|No| F20[result = spoof]
    F19 --> F21[Create Result Dictionary<br/>file, prediction, confidence,<br/>bonafide_prob, spoof_prob]
    F20 --> F21
    F21 --> F22[Append to test_results]
    F22 --> F23[Print Result]
    F23 --> F24[Next File]
    F24 --> F5
    
    F25 --> END[End: Return Results]
    Z1 --> END
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef augment fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef nes2net fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef validation fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    
    class A,END startEnd
    class B,C,G,H,I,J,K,R1,R2,R3,T1,T2,T3,T4,T5,U1,U2,U3,U4,U5,F1,F2,F3,F7,F8,F12,F14,F15,F16,F17,F21,F22,F23 process
    class D,M,P,Q4,Q8,Q10,Q12,S4,S10,S11,S13,S12C,S19,V2,V6,V17,F5,F9,F18 decision
    class Q1,Q2,Q3,Q5,Q6,Q7,Q9,Q11,Q13 augment
    class S1,S2,S3,S5,S6,S7,S8,S9,S12,S12A,S12B,S12D,S14,S15,S16,S17,S18,S19A,S19B,S19C,S19D,S19E,S19F,S20,F13 nes2net
    class V1,V3,V4,V5,V7,V8,V9,V10,V11,V12,V13,V15,V16,V18,V19,V20 validation`,
      pseudocode: `ALGORITHM: Audio_Spoof_Detection_System
INPUT: mode, audio_data, model_path (optional)
OUTPUT: trained_model (if training) OR classification_result (if testing)

BEGIN Audio_Spoof_Detection_System
    
    // ========== INITIALIZATION ==========
    INITIALIZE global_config {
        sample_rate: 16000,
        max_audio_length: 64000,
        ssl_feature_dim: 1024,
        nes2net_groups: 4,
        scale_factor: 4,
        learning_rate: 0.001,
        batch_size: 32,
        max_epochs: 100,
        augmentation_prob: 0.7,
        validation_freq: 10
    }
    
    // Load foundation model (WavLM/Wav2Vec2)
    foundation_model ← LOAD_PRETRAINED_SSL_MODEL()
    
    IF mode == "TRAIN" THEN
        GOTO TRAINING_MODE
    ELSE IF mode == "TEST" THEN
        GOTO TESTING_MODE
    END IF
    
    // ========== TRAINING MODE ==========
    TRAINING_MODE:
        PRINT "Starting Training Pipeline..."
        
        // Data Preparation
        train_dataset ← LOAD_TRAINING_DATA()
        val_dataset ← LOAD_VALIDATION_DATA()
        
        // Model Initialization
        nes2net_model ← INITIALIZE_NES2NET(global_config.ssl_feature_dim, 
                                          global_config.nes2net_groups, 
                                          global_config.scale_factor)
        
        classifier ← INITIALIZE_CLASSIFIER(global_config.ssl_feature_dim)
        optimizer ← ADAM_OPTIMIZER(nes2net_model.parameters + classifier.parameters, 
                                  global_config.learning_rate)
        loss_function ← CROSS_ENTROPY_LOSS()
        
        best_accuracy ← 0.0
        
        // Training Loop
        FOR epoch = 1 TO global_config.max_epochs DO
            PRINT "Epoch:", epoch
            total_loss ← 0.0
            nes2net_model.TRAIN_MODE()
            classifier.TRAIN_MODE()
            
            FOR each batch IN train_dataset DO
                // ===== AUDIO PREPROCESSING & AUGMENTATION =====
                batch_audio ← []
                FOR each audio_file IN batch.audio_files DO
                    // Load and preprocess audio
                    raw_audio ← LOAD_AUDIO(audio_file, global_config.sample_rate)
                    normalized_audio ← NORMALIZE_AMPLITUDE(raw_audio)
                    trimmed_audio ← TRIM_SILENCE(normalized_audio)
                    
                    // Fixed length processing
                    IF LENGTH(trimmed_audio) > global_config.max_audio_length THEN
                        fixed_audio ← RANDOM_CROP(trimmed_audio, global_config.max_audio_length)
                    ELSE
                        fixed_audio ← PAD_ZEROS(trimmed_audio, global_config.max_audio_length)
                    END IF
                    
                    // Apply augmentations
                    augmented_audio ← fixed_audio
                    
                    // RawBoost Augmentation
                    IF RANDOM() < global_config.augmentation_prob THEN
                        rawboost_type ← RANDOM_INTEGER(1, 7)
                        SWITCH rawboost_type:
                            CASE 1: augmented_audio ← LINEAR_FILTER(augmented_audio)
                            CASE 2: augmented_audio ← ADD_NOISE(augmented_audio)
                            CASE 3: augmented_audio ← ADD_REVERB(augmented_audio)
                            CASE 4: augmented_audio ← SCALE_SIGNAL(augmented_audio)
                            CASE 5: augmented_audio ← ADD_ECHO(augmented_audio)
                            CASE 6: augmented_audio ← APPLY_CLIPPING(augmented_audio)
                            CASE 7: augmented_audio ← APPLY_COMPRESSION(augmented_audio)
                        END SWITCH
                    END IF
                    
                    // MUSAN Noise Addition
                    IF RANDOM() < global_config.augmentation_prob THEN
                        noise_type ← RANDOM_CHOICE(["music", "speech", "noise"])
                        noise_file ← RANDOM_SELECT_FROM_MUSAN(noise_type)
                        noise_signal ← LOAD_AUDIO(noise_file, global_config.sample_rate)
                        snr_db ← RANDOM_UNIFORM(5, 20)
                        augmented_audio ← MIX_AUDIO_WITH_NOISE(augmented_audio, noise_signal, snr_db)
                    END IF
                    
                    // Speed Perturbation
                    IF RANDOM() < global_config.augmentation_prob THEN
                        speed_factor ← RANDOM_UNIFORM(0.9, 1.1)
                        augmented_audio ← CHANGE_SPEED(augmented_audio, speed_factor)
                    END IF
                    
                    batch_audio.APPEND(augmented_audio)
                END FOR
                
                // ===== FEATURE EXTRACTION =====
                model_input ← CONVERT_TO_TENSOR(batch_audio)
                WITH torch.no_grad():
                    ssl_features ← foundation_model.FORWARD(model_input)
                    // ssl_features shape: [batch_size, time_steps, feature_dim]
                END WITH
                
                // ===== NES2NET PROCESSING =====
                features ← TRANSPOSE(ssl_features, dim1=1, dim2=2)  // [batch, feature_dim, time]
                nested_outputs ← []
                
                FOR group_idx = 0 TO global_config.nes2net_groups-1 DO
                    // Nested Res2Net Block
                    residual ← features
                    width ← global_config.ssl_feature_dim / global_config.scale_factor
                    feature_splits ← SPLIT(features, width, dim=1)
                    scale_outputs ← []
                    
                    FOR scale_idx = 0 TO global_config.scale_factor-1 DO
                        IF scale_idx == 0 THEN
                            scale_out ← feature_splits[scale_idx]
                        ELIF scale_idx == 1 THEN
                            scale_out ← RELU(CONV1D(feature_splits[scale_idx], width, kernel=3))
                        ELSE
                            nested_input ← feature_splits[scale_idx] + scale_outputs[scale_idx-2]
                            scale_out ← RELU(CONV1D(nested_input, width, kernel=3))
                            IF scale_idx < global_config.scale_factor-1 THEN
                                scale_out ← CONV1D(scale_out, width, kernel=1) + scale_out
                            END IF
                        END IF
                        scale_outputs.APPEND(scale_out)
                    END FOR
                    
                    concatenated ← CONCATENATE(scale_outputs, dim=1)
                    normalized ← BATCH_NORM(concatenated)
                    nested_out ← normalized + residual
                    
                    // Group convolution
                    group_out ← CONV1D(features, global_config.ssl_feature_dim, kernel=3)
                    
                    // Feature fusion
                    IF group_idx > 0 THEN
                        combined ← CONCATENATE([nested_out, nested_outputs[group_idx-1]], dim=1)
                        fused ← CONV1D(combined, global_config.ssl_feature_dim, kernel=1)
                        nested_outputs.APPEND(fused)
                    ELSE
                        nested_outputs.APPEND(nested_out)
                    END IF
                    
                    features ← nested_outputs[group_idx]
                END FOR
                
                // Global pooling
                pooled_features ← GLOBAL_AVERAGE_POOL(features, dim=2)  // [batch, feature_dim]
                
                // ===== CLASSIFICATION =====
                hidden1 ← RELU(LINEAR(pooled_features, 256))
                dropout1 ← DROPOUT(hidden1, 0.3)
                hidden2 ← RELU(LINEAR(dropout1, 64))
                dropout2 ← DROPOUT(hidden2, 0.3)
                logits ← LINEAR(dropout2, 2)  // Binary classification
                
                // ===== LOSS COMPUTATION AND BACKPROPAGATION =====
                loss ← loss_function(logits, batch.labels)
                total_loss ← total_loss + loss
                
                optimizer.ZERO_GRAD()
                loss.BACKWARD()
                optimizer.STEP()
            END FOR
            
            // ===== VALIDATION =====
            IF epoch MOD global_config.validation_freq == 0 THEN
                nes2net_model.EVAL_MODE()
                classifier.EVAL_MODE()
                
                correct_predictions ← 0
                total_samples ← 0
                
                FOR each val_batch IN val_dataset DO
                    val_audio ← []
                    FOR each audio_file IN val_batch.audio_files DO
                        // Preprocess without augmentation
                        raw_audio ← LOAD_AUDIO(audio_file, global_config.sample_rate)
                        normalized_audio ← NORMALIZE_AMPLITUDE(raw_audio)
                        trimmed_audio ← TRIM_SILENCE(normalized_audio)
                        
                        IF LENGTH(trimmed_audio) > global_config.max_audio_length THEN
                            fixed_audio ← CENTER_CROP(trimmed_audio, global_config.max_audio_length)
                        ELSE
                            fixed_audio ← PAD_ZEROS(trimmed_audio, global_config.max_audio_length)
                        END IF
                        
                        val_audio.APPEND(fixed_audio)
                    END FOR
                    
                    // Feature extraction and processing (same as training but no augmentation)
                    val_input ← CONVERT_TO_TENSOR(val_audio)
                    WITH torch.no_grad():
                        val_ssl_features ← foundation_model.FORWARD(val_input)
                        
                        // Nes2Net processing (same logic as training)
                        val_features ← TRANSPOSE(val_ssl_features, dim1=1, dim2=2)
                        val_nested_outputs ← []
                        
                        FOR group_idx = 0 TO global_config.nes2net_groups-1 DO
                            // [Same Nes2Net processing logic as training]
                            val_nested_out ← NESTED_RES2NET_BLOCK(val_features, group_idx)
                            val_group_out ← GROUP_CONV(val_features, group_idx)
                            
                            IF group_idx > 0 THEN
                                val_combined ← CONCATENATE([val_nested_out, val_nested_outputs[group_idx-1]], dim=1)
                                val_fused ← FUSION_CONV(val_combined, group_idx)
                                val_nested_outputs.APPEND(val_fused)
                            ELSE
                                val_nested_outputs.APPEND(val_nested_out)
                            END IF
                            
                            val_features ← val_nested_outputs[group_idx]
                        END FOR
                        
                        val_pooled ← GLOBAL_AVERAGE_POOL(val_features, dim=2)
                        val_hidden1 ← RELU(LINEAR(val_pooled, 256))
                        val_hidden2 ← RELU(LINEAR(val_hidden1, 64))
                        val_logits ← LINEAR(val_hidden2, 2)
                        
                        val_predictions ← ARGMAX(val_logits, dim=1)
                        correct_predictions ← correct_predictions + SUM(val_predictions == val_batch.labels)
                        total_samples ← total_samples + BATCH_SIZE(val_batch)
                    END WITH
                END FOR
                
                validation_accuracy ← (correct_predictions / total_samples) * 100
                PRINT "Epoch:", epoch, "Loss:", total_loss, "Validation Accuracy:", validation_accuracy
                
                // Model checkpointing
                IF validation_accuracy > best_accuracy THEN
                    SAVE_MODEL(nes2net_model, "best_nes2net.pth")
                    SAVE_MODEL(classifier, "best_classifier.pth")
                    best_accuracy ← validation_accuracy
                    PRINT "New best model saved with accuracy:", best_accuracy
                END IF
            END IF
        END FOR
        
        PRINT "Training completed. Best validation accuracy:", best_accuracy
        RETURN nes2net_model, classifier
    
    // ========== TESTING MODE ==========
    TESTING_MODE:
        PRINT "Starting Testing Pipeline..."
        
        // Load trained models
        nes2net_model ← LOAD_MODEL("best_nes2net.pth")
        classifier ← LOAD_MODEL("best_classifier.pth")
        nes2net_model.EVAL_MODE()
        classifier.EVAL_MODE()
        
        test_results ← []
        
        FOR each test_audio_file IN audio_data DO
            // ===== AUDIO PREPROCESSING (NO AUGMENTATION) =====
            raw_audio ← LOAD_AUDIO(test_audio_file, global_config.sample_rate)
            normalized_audio ← NORMALIZE_AMPLITUDE(raw_audio)
            trimmed_audio ← TRIM_SILENCE(normalized_audio)
            
            IF LENGTH(trimmed_audio) > global_config.max_audio_length THEN
                fixed_audio ← CENTER_CROP(trimmed_audio, global_config.max_audio_length)
            ELSE
                fixed_audio ← PAD_ZEROS(trimmed_audio, global_config.max_audio_length)
            END IF
            
            // ===== FEATURE EXTRACTION =====
            test_input ← CONVERT_TO_TENSOR([fixed_audio])
            WITH torch.no_grad():
                ssl_features ← foundation_model.FORWARD(test_input)
                
                // ===== NES2NET PROCESSING =====
                features ← TRANSPOSE(ssl_features, dim1=1, dim2=2)
                nested_outputs ← []
                
                FOR group_idx = 0 TO global_config.nes2net_groups-1 DO
                    // Nested Res2Net Block processing
                    residual ← features
                    width ← global_config.ssl_feature_dim / global_config.scale_factor
                    feature_splits ← SPLIT(features, width, dim=1)
                    scale_outputs ← []
                    
                    FOR scale_idx = 0 TO global_config.scale_factor-1 DO
                        IF scale_idx == 0 THEN
                            scale_out ← feature_splits[scale_idx]
                        ELIF scale_idx == 1 THEN
                            scale_out ← RELU(CONV1D(feature_splits[scale_idx], width, kernel=3))
                        ELSE
                            nested_input ← feature_splits[scale_idx] + scale_outputs[scale_idx-2]
                            scale_out ← RELU(CONV1D(nested_input, width, kernel=3))
                            IF scale_idx < global_config.scale_factor-1 THEN
                                scale_out ← CONV1D(scale_out, width, kernel=1) + scale_out
                            END IF
                        END IF
                        scale_outputs.APPEND(scale_out)
                    END FOR
                    
                    concatenated ← CONCATENATE(scale_outputs, dim=1)
                    normalized ← BATCH_NORM(concatenated)
                    nested_out ← normalized + residual
                    
                    group_out ← CONV1D(features, global_config.ssl_feature_dim, kernel=3)
                    
                    IF group_idx > 0 THEN
                        combined ← CONCATENATE([nested_out, nested_outputs[group_idx-1]], dim=1)
                        fused ← CONV1D(combined, global_config.ssl_feature_dim, kernel=1)
                        nested_outputs.APPEND(fused)
                    ELSE
                        nested_outputs.APPEND(nested_out)
                    END IF
                    
                    features ← nested_outputs[group_idx]
                END FOR
                
                // Global pooling and classification
                pooled_features ← GLOBAL_AVERAGE_POOL(features, dim=2)
                hidden1 ← RELU(LINEAR(pooled_features, 256))
                hidden2 ← RELU(LINEAR(hidden1, 64))
                logits ← LINEAR(hidden2, 2)
                
                // ===== RESULT GENERATION =====
                probabilities ← SOFTMAX(logits)
                predicted_class ← ARGMAX(probabilities)
                confidence ← MAX(probabilities)
                
                IF predicted_class == 0 THEN
                    result_label ← "bonafide"
                ELSE
                    result_label ← "spoof"
                END IF
                
                test_result ← {
                    "file": test_audio_file,
                    "prediction": result_label,
                    "confidence": confidence,
                    "bonafide_prob": probabilities[0],
                    "spoof_prob": probabilities[1]
                }
                
                test_results.APPEND(test_result)
                PRINT "File:", test_audio_file, "Prediction:", result_label, "Confidence:", confidence
            END WITH
        END FOR
        
        RETURN test_results

END Audio_Spoof_Detection_System`,
      insights: 'The key insight was that Self-Supervised Learning (SSL) models pre-trained on vast amounts of unlabeled speech data could capture subtle, low-level features indicative of spoofing artifacts that traditional methods miss.'
    }
  },
  {
    title: 'Automatic Speaker Verification System',
    description: 'Built an end-to-end biometric speaker-verification pipeline using deep speaker embeddings and ensemble score fusion. Trained on 350h of multilingual speech, trimming tandem EER to 30%.',
    technologies: ['Python', 'PyTorch', 'SpeechBrain', 'HuggingFace Transformers', 'Docker', 'Linux'],
    // link: 'https://github.com/ayushsaun24024',
    image: '/data/automaticSpeakerVerification.png',
    aiHint: 'voice recognition',
    details: {
        flowchart: `graph TD
    A[Start: Audio Spoof Detection + ASV System] --> B{Initialize Global Config}
    B --> C[Load Foundation Models<br/>SSL Model + Speaker Model]
    C --> D{Check Mode}
    
    D -->|TRAIN| E[Joint Training Pipeline]
    D -->|ENROLL| F[Speaker Enrollment Pipeline]
    D -->|VERIFY| G[Two-Stage Verification Pipeline]
    
    %% Training Pipeline
    E --> E1[Load Spoof Training Data]
    E1 --> E2[Load Speaker Training Data]
    E2 --> E3[Load Validation Datasets]
    E3 --> E4[Initialize Models<br/>Nes2Net + Spoof Classifier<br/>Speaker Backbone + Speaker Classifier]
    E4 --> E5[Initialize Optimizers & Loss Functions<br/>CrossEntropy + AAM-Softmax]
    E5 --> E6[Set best_accuracies = 0]
    E6 --> E7[Start Epoch Loop]
    
    E7 --> E8{epoch <= max_epochs?}
    E8 -->|Yes| E9[Set Models to Train Mode]
    E8 -->|No| E50[Training Complete<br/>Return All Models]
    
    E9 --> E10[Start Spoof Training Batch Loop]
    E10 --> E11{More spoof batches?}
    E11 -->|Yes| E12[Preprocess & Augment Audio<br/>RawBoost + MUSAN + Speed]
    E11 -->|No| E20[Start Speaker Training]
    
    E12 --> E13[Extract SSL Features]
    E13 --> E14[Process Through Nes2Net]
    E14 --> E15[Spoof Classification]
    E15 --> E16[Compute Spoof Loss]
    E16 --> E17[Spoof Backpropagation]
    E17 --> E18[Update Spoof Models]
    E18 --> E11
    
    E20 --> E21{More speaker batches?}
    E21 -->|Yes| E22[Preprocess & Augment Audio<br/>Same as Spoof Training]
    E21 -->|No| E30[Validation Check]
    
    E22 --> E23[Extract Speaker Embeddings]
    E23 --> E24[Speaker Classification]
    E24 --> E25[Compute Speaker Loss<br/>AAM-Softmax]
    E25 --> E26[Speaker Backpropagation]
    E26 --> E27[Update Speaker Models]
    E27 --> E21
    
    E30 --> E31{epoch % validation_freq == 0?}
    E31 -->|Yes| E32[Validate Spoof Detection]
    E31 -->|No| E45[Next Epoch]
    
    E32 --> E33[Validate Speaker Verification]
    E33 --> E34[Print Training Results]
    E34 --> E35{spoof_accuracy > best_spoof?}
    E35 -->|Yes| E36[Save Best Spoof Models]
    E35 -->|No| E37{speaker_accuracy > best_speaker?}
    E36 --> E37
    E37 -->|Yes| E38[Save Best Speaker Models]
    E37 -->|No| E45
    E38 --> E45
    E45 --> E7
    
    %% Enrollment Pipeline
    F --> F1[Load Speaker Backbone Model]
    F1 --> F2[Set Model to Eval Mode]
    F2 --> F3[Initialize enrollment_embeddings list]
    F3 --> F4[Start Enrollment Audio Loop]
    
    F4 --> F5{More enrollment audios?}
    F5 -->|Yes| F6[Preprocess Audio<br/>NO AUGMENTATION]
    F5 -->|No| F15[Compute Mean Template]
    
    F6 --> F7[Load & Normalize Audio]
    F7 --> F8[Trim Silence]
    F8 --> F9{Length > max_length?}
    F9 -->|Yes| F10[Center Crop]
    F9 -->|No| F11[Pad with Zeros]
    F10 --> F12[Extract Speaker Embedding]
    F11 --> F12
    F12 --> F13[Append to enrollment_embeddings]
    F13 --> F14[Next Enrollment Audio]
    F14 --> F5
    
    F15 --> F16[target_template = MEAN embeddings]
    F16 --> F17[L2 Normalize Template]
    F17 --> F18[Save Speaker Template]
    F18 --> F19[Enrollment Complete]
    F19 --> END1[End: Return Template]
    
    %% Two-Stage Verification Pipeline
    G --> G1[Load All Trained Models<br/>Nes2Net + Spoof Classifier<br/>Speaker Backbone + Template]
    G1 --> G2[Set Models to Eval Mode]
    G2 --> G3[Initialize verification_results list]
    G3 --> G4[Start Test Audio Loop]
    
    G4 --> G5{More test audios?}
    G5 -->|Yes| G6[STAGE 1: Spoof Detection]
    G5 -->|No| G40[Return Verification Results]
    
    %% Stage 1: Spoof Detection
    G6 --> G7[Preprocess Audio<br/>NO AUGMENTATION]
    G7 --> G8[Load & Normalize Audio]
    G8 --> G9[Trim Silence & Fix Length]
    G9 --> G10[Extract SSL Features]
    G10 --> G11[Process Through Nes2Net<br/>Same as Training]
    G11 --> G12[Spoof Classification]
    G12 --> G13[Apply Softmax<br/>Get Probabilities]
    G13 --> G14[bonafide_score = probabilities 0]
    G14 --> G15{bonafide_score >= spoof_threshold?}
    
    G15 -->|No| G16[Decision: SPOOF<br/>speaker_similarity = 0<br/>speaker_decision = N/A]
    G15 -->|Yes| G17[STAGE 2: Speaker Verification]
    
    G16 --> G35[Create Verification Result<br/>final_decision = SPOOF]
    
    %% Stage 2: Speaker Verification
    G17 --> G18[Extract Test Speaker Embedding]
    G18 --> G19[L2 Normalize Embedding]
    G19 --> G20{similarity_metric == cosine?}
    G20 -->|Yes| G21[Compute Cosine Similarity<br/>with Target Template]
    G20 -->|No| G22[Compute Euclidean Distance<br/>Convert to Similarity]
    G21 --> G23[speaker_similarity = similarity_score]
    G22 --> G23
    
    G23 --> G24{speaker_similarity >= verification_threshold?}
    G24 -->|Yes| G25[speaker_decision = ACCEPT<br/>final_decision = ACCEPT]
    G24 -->|No| G26[speaker_decision = REJECT<br/>final_decision = REJECT]
    
    G25 --> G27[Create Verification Result<br/>Include Both Stages]
    G26 --> G27
    G27 --> G35
    
    G35 --> G36[Append to verification_results]
    G36 --> G37[Print Decision Results<br/>Bonafide Score + Similarity]
    G37 --> G38[Next Test Audio]
    G38 --> G5
    
    G40 --> END2[End: Return All Results]
    E50 --> END2
    
    %% Helper Functions (Referenced but not detailed in main flow)
    H1[HELPER: Preprocess & Augment Audio<br/>RawBoost + MUSAN + Speed Perturbation]
    H2[HELPER: Process Through Nes2Net<br/>Nested Res2Net Blocks + Group Conv + Fusion]
    H3[HELPER: Extract Speaker Embeddings<br/>Speaker Backbone + L2 Normalization]
    H4[HELPER: Validate Models<br/>Compute Accuracies on Validation Sets]
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef training fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef enrollment fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef spoofDetection fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef speakerVerif fill:#e8eaf6,stroke:#283593,stroke-width:2px
    classDef helper fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray: 5 5
    
    class A,END1,END2 startEnd
    class B,C,G1,G2,G3,G7,G8,G9,G35,G36,G37,F1,F2,F3,F6,F7,F8,F12,F13,F15,F16,F17,F18,F19 process
    class D,E8,E11,E21,E31,E35,E37,F5,F9,G5,G15,G20,G24 decision
    class E1,E2,E3,E4,E5,E6,E7,E9,E10,E12,E13,E14,E15,E16,E17,E18,E20,E22,E23,E24,E25,E26,E27,E30,E32,E33,E34,E36,E38,E45,E50 training
    class F4,F10,F11,F14 enrollment
    class G4,G6,G10,G11,G12,G13,G14,G16,G38,G40 spoofDetection
    class G17,G18,G19,G21,G22,G23,G25,G26,G27 speakerVerif
    class H1,H2,H3,H4 helper`,
        pseudocode: `ALGORITHM: Audio_Spoof_Detection_And_Speaker_Verification_System
INPUT: mode, audio_data, target_speaker_data (optional), model_path (optional)
OUTPUT: trained_models (if training) OR verification_result (if testing/enrollment)

BEGIN Audio_Spoof_Detection_And_Speaker_Verification_System
    
    // ========== INITIALIZATION ==========
    INITIALIZE global_config {
        sample_rate: 16000,
        max_audio_length: 64000,
        ssl_feature_dim: 1024,
        speaker_embed_dim: 192,
        nes2net_groups: 4,
        scale_factor: 4,
        learning_rate: 0.001,
        batch_size: 32,
        max_epochs: 100,
        augmentation_prob: 0.7,
        validation_freq: 10,
        spoof_threshold: 0.5,
        speaker_verification_threshold: 0.25,
        similarity_metric: "cosine"
    }
    
    // Load foundation models
    foundation_model ← LOAD_PRETRAINED_SSL_MODEL()  // For spoof detection
    speaker_model ← LOAD_PRETRAINED_SPEAKER_MODEL()  // For speaker verification (ECAPA-TDNN/ResNet)
    
    IF mode == "TRAIN" THEN
        GOTO TRAINING_MODE
    ELSE IF mode == "ENROLL" THEN
        GOTO ENROLLMENT_MODE
    ELSE IF mode == "VERIFY" THEN
        GOTO VERIFICATION_MODE
    END IF
    
    // ========== TRAINING MODE ==========
    TRAINING_MODE:
        PRINT "Starting Joint Training Pipeline..."
        
        // Data Preparation
        spoof_train_dataset ← LOAD_SPOOF_TRAINING_DATA()  // Bonafide/Spoof labels
        spoof_val_dataset ← LOAD_SPOOF_VALIDATION_DATA()
        speaker_train_dataset ← LOAD_SPEAKER_TRAINING_DATA()  // Speaker ID labels
        speaker_val_dataset ← LOAD_SPEAKER_VALIDATION_DATA()
        
        // Model Initialization
        // Spoof Detection Models
        nes2net_model ← INITIALIZE_NES2NET(global_config.ssl_feature_dim, 
                                          global_config.nes2net_groups, 
                                          global_config.scale_factor)
        spoof_classifier ← INITIALIZE_SPOOF_CLASSIFIER(global_config.ssl_feature_dim)
        
        // Speaker Verification Models
        speaker_backbone ← INITIALIZE_SPEAKER_BACKBONE(global_config.speaker_embed_dim)
        speaker_classifier ← INITIALIZE_SPEAKER_CLASSIFIER(global_config.speaker_embed_dim, num_speakers)
        
        // Optimizers
        spoof_optimizer ← ADAM_OPTIMIZER(nes2net_model.parameters + spoof_classifier.parameters, 
                                        global_config.learning_rate)
        speaker_optimizer ← ADAM_OPTIMIZER(speaker_backbone.parameters + speaker_classifier.parameters,
                                          global_config.learning_rate)
        
        // Loss Functions
        spoof_loss_function ← CROSS_ENTROPY_LOSS()
        speaker_loss_function ← AAM_SOFTMAX_LOSS()  // Additive Angular Margin Loss
        
        best_spoof_accuracy ← 0.0
        best_speaker_accuracy ← 0.0
        
        // Joint Training Loop
        FOR epoch = 1 TO global_config.max_epochs DO
            PRINT "Epoch:", epoch
            total_spoof_loss ← 0.0
            total_speaker_loss ← 0.0
            
            // Set models to training mode
            nes2net_model.TRAIN_MODE()
            spoof_classifier.TRAIN_MODE()
            speaker_backbone.TRAIN_MODE()
            speaker_classifier.TRAIN_MODE()
            
            // ===== SPOOF DETECTION TRAINING =====
            FOR each batch IN spoof_train_dataset DO
                // Audio preprocessing and augmentation (same as before)
                batch_audio ← []
                FOR each audio_file IN batch.audio_files DO
                    processed_audio ← PREPROCESS_AND_AUGMENT_AUDIO(audio_file)
                    batch_audio.APPEND(processed_audio)
                END FOR
                
                // Feature extraction and Nes2Net processing
                ssl_features ← EXTRACT_SSL_FEATURES(batch_audio, foundation_model)
                nes2net_features ← PROCESS_THROUGH_NES2NET(ssl_features, nes2net_model)
                spoof_logits ← spoof_classifier(nes2net_features)
                
                // Spoof detection loss and backpropagation
                spoof_loss ← spoof_loss_function(spoof_logits, batch.spoof_labels)
                total_spoof_loss ← total_spoof_loss + spoof_loss
                
                spoof_optimizer.ZERO_GRAD()
                spoof_loss.BACKWARD()
                spoof_optimizer.STEP()
            END FOR
            
            // ===== SPEAKER VERIFICATION TRAINING =====
            FOR each batch IN speaker_train_dataset DO
                // Audio preprocessing and augmentation (same logic)
                batch_audio ← []
                FOR each audio_file IN batch.audio_files DO
                    processed_audio ← PREPROCESS_AND_AUGMENT_AUDIO(audio_file)
                    batch_audio.APPEND(processed_audio)
                END FOR
                
                // Speaker embedding extraction
                speaker_embeddings ← EXTRACT_SPEAKER_EMBEDDINGS(batch_audio, speaker_backbone)
                speaker_logits ← speaker_classifier(speaker_embeddings)
                
                // Speaker verification loss and backpropagation
                speaker_loss ← speaker_loss_function(speaker_logits, batch.speaker_labels)
                total_speaker_loss ← total_speaker_loss + speaker_loss
                
                speaker_optimizer.ZERO_GRAD()
                speaker_loss.BACKWARD()
                speaker_optimizer.STEP()
            END FOR
            
            // ===== VALIDATION =====
            IF epoch MOD global_config.validation_freq == 0 THEN
                // Validate Spoof Detection
                spoof_accuracy ← VALIDATE_SPOOF_DETECTION(spoof_val_dataset, nes2net_model, spoof_classifier)
                
                // Validate Speaker Verification
                speaker_accuracy ← VALIDATE_SPEAKER_VERIFICATION(speaker_val_dataset, speaker_backbone, speaker_classifier)
                
                PRINT "Epoch:", epoch, "Spoof Loss:", total_spoof_loss, "Speaker Loss:", total_speaker_loss
                PRINT "Spoof Accuracy:", spoof_accuracy, "Speaker Accuracy:", speaker_accuracy
                
                // Model checkpointing
                IF spoof_accuracy > best_spoof_accuracy THEN
                    SAVE_MODEL(nes2net_model, "best_nes2net.pth")
                    SAVE_MODEL(spoof_classifier, "best_spoof_classifier.pth")
                    best_spoof_accuracy ← spoof_accuracy
                END IF
                
                IF speaker_accuracy > best_speaker_accuracy THEN
                    SAVE_MODEL(speaker_backbone, "best_speaker_backbone.pth")
                    SAVE_MODEL(speaker_classifier, "best_speaker_classifier.pth")
                    best_speaker_accuracy ← speaker_accuracy
                END IF
            END IF
        END FOR
        
        PRINT "Training completed. Best Spoof Accuracy:", best_spoof_accuracy
        PRINT "Best Speaker Accuracy:", best_speaker_accuracy
        RETURN nes2net_model, spoof_classifier, speaker_backbone, speaker_classifier
    
    // ========== ENROLLMENT MODE ==========
    ENROLLMENT_MODE:
        PRINT "Starting Speaker Enrollment..."
        
        // Load trained models
        speaker_backbone ← LOAD_MODEL("best_speaker_backbone.pth")
        speaker_backbone.EVAL_MODE()
        
        enrollment_embeddings ← []
        
        FOR each enrollment_audio IN target_speaker_data DO
            // Audio preprocessing (no augmentation during enrollment)
            raw_audio ← LOAD_AUDIO(enrollment_audio, global_config.sample_rate)
            normalized_audio ← NORMALIZE_AMPLITUDE(raw_audio)
            trimmed_audio ← TRIM_SILENCE(normalized_audio)
            
            IF LENGTH(trimmed_audio) > global_config.max_audio_length THEN
                fixed_audio ← CENTER_CROP(trimmed_audio, global_config.max_audio_length)
            ELSE
                fixed_audio ← PAD_ZEROS(trimmed_audio, global_config.max_audio_length)
            END IF
            
            // Extract speaker embedding
            WITH torch.no_grad():
                speaker_embedding ← EXTRACT_SPEAKER_EMBEDDINGS([fixed_audio], speaker_backbone)
                enrollment_embeddings.APPEND(speaker_embedding)
            END WITH
        END FOR
        
        // Compute mean enrollment embedding (template)
        target_speaker_template ← MEAN(enrollment_embeddings, dim=0)
        target_speaker_template ← L2_NORMALIZE(target_speaker_template)
        
        // Save enrollment template
        SAVE_TEMPLATE(target_speaker_template, "target_speaker_template.pth")
        PRINT "Speaker enrollment completed. Template saved."
        RETURN target_speaker_template
    
    // ========== VERIFICATION MODE ==========
    VERIFICATION_MODE:
        PRINT "Starting Two-Stage Verification..."
        
        // Load all trained models
        nes2net_model ← LOAD_MODEL("best_nes2net.pth")
        spoof_classifier ← LOAD_MODEL("best_spoof_classifier.pth")
        speaker_backbone ← LOAD_MODEL("best_speaker_backbone.pth")
        target_speaker_template ← LOAD_TEMPLATE("target_speaker_template.pth")
        
        // Set models to evaluation mode
        nes2net_model.EVAL_MODE()
        spoof_classifier.EVAL_MODE()
        speaker_backbone.EVAL_MODE()
        
        verification_results ← []
        
        FOR each test_audio_file IN audio_data DO
            PRINT "Processing:", test_audio_file
            
            // ===== STAGE 1: SPOOF DETECTION =====
            // Audio preprocessing (no augmentation)
            raw_audio ← LOAD_AUDIO(test_audio_file, global_config.sample_rate)
            normalized_audio ← NORMALIZE_AMPLITUDE(raw_audio)
            trimmed_audio ← TRIM_SILENCE(normalized_audio)
            
            IF LENGTH(trimmed_audio) > global_config.max_audio_length THEN
                fixed_audio ← CENTER_CROP(trimmed_audio, global_config.max_audio_length)
            ELSE
                fixed_audio ← PAD_ZEROS(trimmed_audio, global_config.max_audio_length)
            END IF
            
            WITH torch.no_grad():
                // Spoof detection processing
                ssl_features ← EXTRACT_SSL_FEATURES([fixed_audio], foundation_model)
                nes2net_features ← PROCESS_THROUGH_NES2NET(ssl_features, nes2net_model)
                spoof_logits ← spoof_classifier(nes2net_features)
                spoof_probabilities ← SOFTMAX(spoof_logits)
                
                spoof_prediction ← ARGMAX(spoof_probabilities)
                spoof_confidence ← MAX(spoof_probabilities)
                bonafide_score ← spoof_probabilities[0]  // Probability of being bonafide
                
                // Check spoof threshold
                IF bonafide_score < global_config.spoof_threshold THEN
                    // Audio is detected as spoof
                    final_decision ← "SPOOF"
                    speaker_similarity ← 0.0
                    speaker_decision ← "N/A"
                    
                    verification_result ← {
                        "file": test_audio_file,
                        "final_decision": final_decision,
                        "spoof_detection": {
                            "prediction": "spoof",
                            "bonafide_score": bonafide_score,
                            "confidence": spoof_confidence
                        },
                        "speaker_verification": {
                            "similarity": speaker_similarity,
                            "decision": speaker_decision
                        }
                    }
                ELSE
                    // Audio is bonafide, proceed to speaker verification
                    PRINT "Audio is bonafide, proceeding to speaker verification..."
                    
                    // ===== STAGE 2: SPEAKER VERIFICATION =====
                    test_speaker_embedding ← EXTRACT_SPEAKER_EMBEDDINGS([fixed_audio], speaker_backbone)
                    test_speaker_embedding ← L2_NORMALIZE(test_speaker_embedding)
                    
                    // Compute similarity with target speaker template
                    IF global_config.similarity_metric == "cosine" THEN
                        speaker_similarity ← COSINE_SIMILARITY(test_speaker_embedding, target_speaker_template)
                    ELSE IF global_config.similarity_metric == "euclidean" THEN
                        speaker_similarity ← 1.0 / (1.0 + EUCLIDEAN_DISTANCE(test_speaker_embedding, target_speaker_template))
                    END IF
                    
                    // Speaker verification decision
                    IF speaker_similarity >= global_config.speaker_verification_threshold THEN
                        speaker_decision ← "ACCEPT"
                        final_decision ← "ACCEPT"
                    ELSE
                        speaker_decision ← "REJECT"
                        final_decision ← "REJECT"
                    END IF
                    
                    verification_result ← {
                        "file": test_audio_file,
                        "final_decision": final_decision,
                        "spoof_detection": {
                            "prediction": "bonafide",
                            "bonafide_score": bonafide_score,
                            "confidence": spoof_confidence
                        },
                        "speaker_verification": {
                            "similarity": speaker_similarity,
                            "decision": speaker_decision
                        }
                    }
                END IF
            END WITH
            
            verification_results.APPEND(verification_result)
            PRINT "File:", test_audio_file, "Final Decision:", final_decision
            PRINT "Bonafide Score:", bonafide_score, "Speaker Similarity:", speaker_similarity
        END FOR
        
        RETURN verification_results

    // ========== HELPER FUNCTIONS ==========
    
    FUNCTION PREPROCESS_AND_AUGMENT_AUDIO(audio_file):
        // Load and preprocess audio
        raw_audio ← LOAD_AUDIO(audio_file, global_config.sample_rate)
        normalized_audio ← NORMALIZE_AMPLITUDE(raw_audio)
        trimmed_audio ← TRIM_SILENCE(normalized_audio)
        
        // Fixed length processing
        IF LENGTH(trimmed_audio) > global_config.max_audio_length THEN
            fixed_audio ← RANDOM_CROP(trimmed_audio, global_config.max_audio_length)
        ELSE
            fixed_audio ← PAD_ZEROS(trimmed_audio, global_config.max_audio_length)
        END IF
        
        // Apply augmentations
        augmented_audio ← fixed_audio
        
        // RawBoost Augmentation
        IF RANDOM() < global_config.augmentation_prob THEN
            rawboost_type ← RANDOM_INTEGER(1, 7)
            SWITCH rawboost_type:
                CASE 1: augmented_audio ← LINEAR_FILTER(augmented_audio)
                CASE 2: augmented_audio ← ADD_NOISE(augmented_audio)
                CASE 3: augmented_audio ← ADD_REVERB(augmented_audio)
                CASE 4: augmented_audio ← SCALE_SIGNAL(augmented_audio)
                CASE 5: augmented_audio ← ADD_ECHO(augmented_audio)
                CASE 6: augmented_audio ← APPLY_CLIPPING(augmented_audio)
                CASE 7: augmented_audio ← APPLY_COMPRESSION(augmented_audio)
            END SWITCH
        END IF
        
        // MUSAN Noise Addition
        IF RANDOM() < global_config.augmentation_prob THEN
            noise_type ← RANDOM_CHOICE(["music", "speech", "noise"])
            noise_file ← RANDOM_SELECT_FROM_MUSAN(noise_type)
            noise_signal ← LOAD_AUDIO(noise_file, global_config.sample_rate)
            snr_db ← RANDOM_UNIFORM(5, 20)
            augmented_audio ← MIX_AUDIO_WITH_NOISE(augmented_audio, noise_signal, snr_db)
        END IF
        
        // Speed Perturbation
        IF RANDOM() < global_config.augmentation_prob THEN
            speed_factor ← RANDOM_UNIFORM(0.9, 1.1)
            augmented_audio ← CHANGE_SPEED(augmented_audio, speed_factor)
        END IF
        
        RETURN augmented_audio
    END FUNCTION
    
    FUNCTION PROCESS_THROUGH_NES2NET(ssl_features, nes2net_model):
        // Same Nes2Net processing logic as in original pseudocode
        features ← TRANSPOSE(ssl_features, dim1=1, dim2=2)
        nested_outputs ← []
        
        FOR group_idx = 0 TO global_config.nes2net_groups-1 DO
            // Nested Res2Net Block processing
            residual ← features
            width ← global_config.ssl_feature_dim / global_config.scale_factor
            feature_splits ← SPLIT(features, width, dim=1)
            scale_outputs ← []
            
            FOR scale_idx = 0 TO global_config.scale_factor-1 DO
                IF scale_idx == 0 THEN
                    scale_out ← feature_splits[scale_idx]
                ELIF scale_idx == 1 THEN
                    scale_out ← RELU(CONV1D(feature_splits[scale_idx], width, kernel=3))
                ELSE
                    nested_input ← feature_splits[scale_idx] + scale_outputs[scale_idx-2]
                    scale_out ← RELU(CONV1D(nested_input, width, kernel=3))
                    IF scale_idx < global_config.scale_factor-1 THEN
                        scale_out ← CONV1D(scale_out, width, kernel=1) + scale_out
                    END IF
                END IF
                scale_outputs.APPEND(scale_out)
            END FOR
            
            concatenated ← CONCATENATE(scale_outputs, dim=1)
            normalized ← BATCH_NORM(concatenated)
            nested_out ← normalized + residual
            
            group_out ← CONV1D(features, global_config.ssl_feature_dim, kernel=3)
            
            IF group_idx > 0 THEN
                combined ← CONCATENATE([nested_out, nested_outputs[group_idx-1]], dim=1)
                fused ← CONV1D(combined, global_config.ssl_feature_dim, kernel=1)
                nested_outputs.APPEND(fused)
            ELSE
                nested_outputs.APPEND(nested_out)
            END IF
            
            features ← nested_outputs[group_idx]
        END FOR
        
        pooled_features ← GLOBAL_AVERAGE_POOL(features, dim=2)
        RETURN pooled_features
    END FUNCTION
    
    FUNCTION EXTRACT_SPEAKER_EMBEDDINGS(audio_batch, speaker_backbone):
        // Convert audio to speaker model input format
        speaker_input ← PREPARE_SPEAKER_INPUT(audio_batch)
        
        // Extract speaker embeddings
        speaker_features ← speaker_backbone(speaker_input)
        speaker_embeddings ← L2_NORMALIZE(speaker_features)
        
        RETURN speaker_embeddings
    END FUNCTION

END Audio_Spoof_Detection_And_Speaker_Verification_System`,
        insights: 'Ensemble methods and score calibration were critical for robustness. Fusing scores from different embedding models significantly reduced the Equal Error Rate compared to any single model.'
      }
  },
  {
    title: 'Classic ML-Based Vocoder',
    description: 'Engineered a Mel-spectrogram-to-waveform vocoder using STFT and Griffin-Lim, delivering studio-grade audio with a Mean Opinion Score (MOS) of 4.1. Streamlined TensorFlow inference for 30% lower latency.',
    technologies: ['Python', 'TensorFlow', 'Librosa', 'STFT', 'Griffin-Lim', 'Docker'],
    link: 'https://github.com/ayushsaun24024/vocoder',
    image: '/data/vocoder.png',
    aiHint: 'sound engineering',
    details: {
        flowchart: `flowchart TD
    A[Start: Main Execution] --> B[Initialize AudioProcessingPipeline]
    B --> B1[Set original_folder, output_folder, n_mels=500]
    B1 --> B2[Create subdirectory paths: npy, wav, png]
    B2 --> B3[Create directories if they don't exist]
    B3 --> B4[Initialize empty results list]
    
    B4 --> C[Call process_audio_files method]
    C --> C1[Get sorted list of original files]
    C1 --> C2{For each .wav file in original_folder}
    
    C2 --> C3[Call compute_mel_spectrogram method]
    C3 --> C3a[Load audio using librosa.load]
    C3a --> C3b[Compute mel-spectrogram with n_mels bands]
    C3b --> C3c[Convert power to dB scale]
    C3c --> C3d[Return mel_spec and sr]
    
    C3d --> C4[Save mel-spectrogram as .npy file]
    C4 --> C5[Call save_mel_plot method]
    C5 --> C5a[Create matplotlib figure]
    C5a --> C5b[Display mel-spectrogram with axes]
    C5b --> C5c[Save plot as PNG file]
    C5c --> C5d[Close plot]
    
    C5d --> C6[Call griffin_lim_reconstruction method]
    C6 --> C6a[Convert mel-spectrogram from dB to power]
    C6a --> C6b[Convert mel to linear spectrogram]
    C6b --> C6c[Apply Griffin-Lim algorithm]
    C6c --> C6d[Return reconstructed audio]
    
    C6d --> C7[Save reconstructed audio as .wav file]
    C7 --> C8{More files to process?}
    C8 -->|Yes| C2
    C8 -->|No| D[Call compare_audio_files method]
    
    D --> D1[Get sorted list of original files]
    D1 --> D2[Get sorted list of generated files]
    D2 --> D3[Initialize audio_results and mel_results lists]
    D3 --> D4{For each original .wav file}
    
    D4 --> D5[Find matching generated file by filename]
    D5 --> D6{Matching file found?}
    D6 -->|No| D7[Print no matching file message]
    D7 --> D18{More original files?}
    
    D6 -->|Yes| D8[Load original and generated audio]
    D8 --> D9[Resample to target_sr=22050 if needed]
    D9 --> D10[Trim both audio to same length]
    
    D10 --> D11[Compute euclidean distance for audio]
    D11 --> D12[Call normalize method for both audio signals]
    D12 --> D12a[Take absolute values]
    D12a --> D12b[Divide by sum to create probability distribution]
    D12b --> D13[Call kl_divergence method for audio]
    D13 --> D13a[Clip values to avoid log of zero]
    D13a --> D13b[Compute sum of p * log of p/q]
    D13b --> D14[Store audio comparison results]
    
    D14 --> D15[Compute mel-spectrograms for both files]
    D15 --> D16[Trim mel-spectrograms to same time frames]
    D16 --> D17[Flatten and compute euclidean distance]
    D17 --> D17a[Normalize flattened mel-spectrograms]
    D17a --> D17b[Compute KL divergence for mel-spectrograms]
    D17b --> D17c[Store mel-spectrogram comparison results]
    
    D17c --> D18{More original files?}
    D18 -->|Yes| D4
    D18 -->|No| D19[Store results in class attributes]
    
    D19 --> E[Call save_results method]
    E --> E1[Convert audio results to DataFrame]
    E1 --> E2[Save audio comparison to CSV]
    E2 --> E3[Convert mel results to DataFrame]
    E3 --> E4[Save mel comparison to CSV]
    E4 --> E5[Print confirmation messages]
    E5 --> F[End]`,
        pseudocode: `CLASS AudioProcessingPipeline:
    
    CONSTRUCTOR(original_folder, output_folder, n_mels=500):
        SET original_folder, output_folder, n_mels
        CREATE subdirectory paths for npy, wav, png outputs
        CREATE directories if they don't exist
        INITIALIZE empty results list
    
    FUNCTION compute_mel_spectrogram(file_path, sr=22050):
        LOAD audio file using librosa
        COMPUTE mel-spectrogram with n_mels bands
        CONVERT power to decibel scale
        RETURN mel_spectrogram, sample_rate
    
    FUNCTION griffin_lim_reconstruction(mel_spec, sr, n_iter=32):
        CONVERT mel-spectrogram from dB to power
        CONVERT mel-spectrogram to linear spectrogram
        APPLY Griffin-Lim algorithm for audio reconstruction
        RETURN reconstructed_audio
    
    FUNCTION normalize(data):
        TAKE absolute values of data
        DIVIDE by sum to create probability distribution
        RETURN normalized_data
    
    FUNCTION kl_divergence(p, q):
        CLIP values to avoid log(0)
        COMPUTE KL divergence: sum(p * log(p/q))
        RETURN kl_divergence_value
    
    FUNCTION process_audio_files():
        FOR each .wav file in original_folder:
            COMPUTE mel-spectrogram
            SAVE mel-spectrogram as .npy file
            SAVE mel-spectrogram plot as .png file
            RECONSTRUCT audio using Griffin-Lim
            SAVE reconstructed audio as .wav file
    
    FUNCTION save_mel_plot(mel_spec, sr, file_name):
        CREATE matplotlib figure
        DISPLAY mel-spectrogram with time/frequency axes
        ADD colorbar and title
        SAVE plot as PNG file
        CLOSE plot
    
    FUNCTION compare_audio_files():
        GET list of original files
        GET list of generated/reconstructed files
        
        FOR each original file:
            FIND matching generated file by filename
            LOAD both original and generated audio
            RESAMPLE to common sample rate if needed
            TRIM to same length
            
            // Audio comparison
            COMPUTE euclidean distance between audio signals
            NORMALIZE both audio signals
            COMPUTE KL divergence
            STORE audio comparison results
            
            // Mel-spectrogram comparison
            COMPUTE mel-spectrograms for both files
            TRIM to same time frames
            FLATTEN mel-spectrograms
            COMPUTE euclidean distance between mel-spectrograms
            NORMALIZE mel-spectrograms
            COMPUTE KL divergence
            STORE mel-spectrogram comparison results
    
    FUNCTION save_results():
        CONVERT audio results to DataFrame
        SAVE audio comparison results to CSV
        CONVERT mel-spectrogram results to DataFrame
        SAVE mel-spectrogram comparison results to CSV
        PRINT confirmation messages

MAIN EXECUTION:
    SET original_folder path
    SET output_folder path
    CREATE AudioProcessingPipeline instance with n_mels=500
    CALL process_audio_files()
    CALL compare_audio_files()
    CALL save_results()`,
        insights: 'The Griffin-Lim algorithm is an iterative process that can reconstruct a time-domain signal from only the magnitude of its Short-Time Fourier Transform (STFT), making it a powerful tool for vocoding tasks when phase information is lost.'
      }
  },
  {
    title: 'Single-Object Tracking System',
    description: 'Delivered a real-time 30 FPS tracker with camera-motion compensation. Integrated hybrid descriptors and ensemble regressors to achieve 85% IoU and 20% lower MAE.',
    technologies: ['Python', 'OpenCV', 'HOG', 'SIFT', 'Random Forest', 'Linear Regression'],
    link: 'https://github.com/ayushsaun24024/Single-Object-Tracking',
    image: '/data/singleObjectTracking.png',
    aiHint: 'object tracking',
    details: {
        flowchart: `graph TD
    A[Start: Initialize Pipeline] --> B[Load Directory Paths]
    B --> C[Initialize Components]
    C --> D[Create Position & Size Scalers]
    D --> E[Create Linear Regression & Random Forest Models]
    E --> F[Initialize Feature Cache & Trackers]
    F --> G[Initialize Camera Motion Compensator]
    G --> H[Get Sequence Folders]
    H --> I[Split into Train/Test Sets]
    
    I --> J[Process Training Sequences]
    J --> K{For Each Training Sequence}
    K --> L[Load Annotations & Images]
    L --> M{For Each Frame in Sequence}
    
    M --> N{Features Cached?}
    N -->|Yes| O[Load Cached Features]
    N -->|No| P[Load Image]
    P --> Q[Estimate Camera Motion with ORB]
    Q --> R{Previous BBox Exists?}
    
    R -->|Yes| S[Generate Multi-scale Windows]
    S --> T{Template Initialized?}
    T -->|No| U[Extract Template from Previous BBox]
    U --> V[Compute SIFT Features for Template]
    V --> W[Score Each Window Against Template]
    T -->|Yes| W
    W --> X[Select Best Scoring Window]
    X --> Y[Extract ROI from Best Window]
    
    R -->|No| Z[Use Entire Image as ROI]
    Y --> AA[Resize ROI to 64x64]
    Z --> AA
    AA --> BB[Extract HOG Features]
    BB --> CC[Compute Local Binary Pattern Features]
    CC --> DD[Add Motion Features from Transform Matrix]
    DD --> EE[Add Position & Size Information]
    EE --> FF[Cache Features]
    FF --> GG[Update Template Counter]
    
    GG --> HH{Template Update Needed?}
    HH -->|Yes| II[Calculate IoU with Ground Truth]
    II --> JJ{"IoU > 0.6?"}
    JJ -->|Yes| KK[Update Template with Current BBox]
    KK --> LL[Reset Template Counter]
    LL --> MM[Store Features & Labels]
    JJ -->|No| MM
    HH -->|No| MM
    
    O --> MM
    MM --> NN{More Frames?}
    NN -->|Yes| M
    NN -->|No| OO{More Training Sequences?}
    OO -->|Yes| K
    OO -->|No| PP[Process Test Sequences]
    
    PP --> QQ{For Each Test Sequence}
    QQ --> RR[Similar Frame Processing as Training]
    RR --> SS{More Test Sequences?}
    SS -->|Yes| QQ
    SS -->|No| TT[Stack All Training Features & Labels]
    
    TT --> UU[Scale Features for Position Model]
    UU --> VV[Scale Features for Size Model]
    VV --> WW["Split Labels: Position x,y & Size w,h"]
    WW --> XX[Train Linear Regression for Position]
    XX --> YY[Train Random Forest for Size]
    YY --> ZZ[Create Output Directory]
    
    ZZ --> AAA{For Each Test Sequence}
    AAA --> BBB[Transform Features with Scalers]
    BBB --> CCC[Predict Positions with Linear Regression]
    CCC --> DDD[Predict Sizes with Random Forest]
    DDD --> EEE[Combine Position & Size Predictions]
    EEE --> FFF[Create Tracking Video]
    
    FFF --> GGG[Initialize Video Writer]
    GGG --> HHH{For Each Frame in Video}
    HHH --> III[Load Current Image]
    III --> JJJ[Estimate Camera Motion for Visualization]
    JJJ --> KKK{Not First Frame?}
    KKK -->|Yes| LLL[Generate & Draw Search Windows]
    LLL --> MMM[Draw Motion Vectors on Grid]
    MMM --> NNN["Draw Ground Truth BBox (Green)"]
    KKK -->|No| NNN
    NNN --> OOO["Draw Predicted BBox (Red)"]
    OOO --> PPP[Calculate & Display IoU]
    PPP --> QQQ[Display Frame Number]
    QQQ --> RRR[Write Frame to Video]
    RRR --> SSS{More Frames?}
    SSS -->|Yes| HHH
    SSS -->|No| TTT[Release Video Writer]
    
    TTT --> UUU{More Test Sequences?}
    UUU -->|Yes| AAA
    UUU -->|No| VVV[Calculate Evaluation Metrics]
    
    VVV --> WWW["Calculate Position Metrics: MAE, RMSE, R²"]
    WWW --> XXX["Calculate Size Metrics: MAE, RMSE, R²"]
    XXX --> YYY[Calculate Overall IoU]
    YYY --> ZZZ[Display Comprehensive Results]
    ZZZ --> AAAA[Return Evaluation Metrics]
    AAAA --> BBBB[End: Pipeline Completed]
    
    style A fill:#e1f5fe
    style BBBB fill:#c8e6c9
    style XX fill:#fff3e0
    style YY fill:#fff3e0
    style VVV fill:#f3e5f5
    style FFF fill:#e8f5e8`,
        pseudocode: `CLASS CameraMotionCompensator:
    INITIALIZE:
        - prev_frame = None
        - prev_keypoints = None
        - prev_descriptors = None
        - ORB feature detector (1000 features)
        - Brute Force matcher with Hamming distance
    
    METHOD estimate_motion(frame):
        IF frame is None:
            RETURN identity transformation matrix
        
        - Convert frame to grayscale
        - Detect ORB keypoints and descriptors
        
        IF this is first frame:
            - Store current frame data
            - RETURN identity matrix
        
        IF insufficient descriptors (< 4):
            RETURN identity matrix
        
        - Match descriptors between previous and current frame
        - IF insufficient matches (< 4):
            RETURN identity matrix
        
        - Sort matches by distance quality
        - Select best 50 matches
        - Extract source and destination points
        - Estimate affine transformation matrix
        - Update previous frame data
        RETURN transformation matrix

CLASS ImprovedSlidingWindowTracker:
    INITIALIZE:
        - scale_factor = 2.0
        - overlap = 0.3
        - SIFT feature detector (2000 features)
        - FLANN matcher for fast nearest neighbor search
        - scale_levels = 3, scale_step = 1.2
    
    METHOD generate_multiscale_windows(img_shape, prev_bbox, transform_matrix):
        - Extract bounding box coordinates (x, y, w, h)
        
        IF transform_matrix provided:
            - Apply camera motion compensation to bbox center
            - Update bbox position based on estimated motion
        
        - Initialize empty windows list
        
        FOR each scale level:
            - Calculate window dimensions with scaling
            - Calculate center position
            - Calculate step sizes based on overlap
            
            FOR each displacement in y-direction:
                FOR each displacement in x-direction:
                    - Calculate window position
                    - Ensure window stays within image bounds
                    - Add window to list
        
        RETURN windows list
    
    METHOD score_window(img, window, template, template_kp, template_desc):
        - Extract region of interest from window
        - Check minimum size requirements
        - Resize ROI to match template size
        - Detect SIFT keypoints and descriptors
        
        IF insufficient descriptors:
            RETURN score = 0
        
        TRY:
            - Match descriptors using FLANN matcher
            - Filter good matches using ratio test (0.7 threshold)
            - Calculate score based on match quantity and quality
            RETURN normalized score
        EXCEPT:
            RETURN score = 0

CLASS ImprovedHybridTrackingPipeline:
    INITIALIZE:
        - Set directory paths for sequences and annotations
        - Initialize scalers for position and size features
        - Create Linear Regression model for position prediction
        - Create Random Forest model for size prediction
        - Initialize feature cache, window tracker, motion compensator
        - Initialize template storage variables
    
    METHOD extract_enhanced_features(img, prev_bbox, transform_matrix):
        IF image is None:
            RETURN None
        
        - Convert image to grayscale
        - Initialize features list
        
        IF previous bounding box exists:
            - Generate multiscale search windows with motion compensation
            - IF template not initialized:
                - Extract template from previous bbox
                - Compute template features
            
            - Score each window against template
            - Select best scoring window
            - Use best window or fallback to previous bbox
        ELSE:
            - Use entire image as ROI
        
        - Resize ROI to standard size (64x64)
        - Extract HOG features (first 64 components)
        - Compute Local Binary Pattern features
        - Add motion features from transformation matrix
        - Add position and size information
        
        RETURN feature vector
    
    METHOD _local_binary_pattern(image, n_points, radius):
        - Initialize output matrix
        FOR each pixel (avoiding borders):
            - Get center pixel value
            - Initialize pattern = 0
            FOR each neighbor point:
                - Calculate neighbor coordinates using circular sampling
                - Perform bilinear interpolation for sub-pixel values
                - Compare neighbor with center pixel
                - Update binary pattern
            - Store pattern in output matrix
        RETURN LBP features
    
    METHOD process_sequence(sequence):
        - Load annotation file for sequence
        - Get sorted list of image files
        - Initialize storage for features, labels, paths
        - Initialize tracking variables
        
        FOR each image and annotation pair:
            IF image cached:
                - Use cached features
            ELSE:
                - Load image
                - Estimate camera motion
                - Extract enhanced features
                - Cache features
                - Update template adaptively based on IoU
            
            - Store features, labels, and paths
            - Update previous bbox
        
        RETURN sequence data dictionary
    
    METHOD calculate_iou(bbox1, bbox2):
        - Extract coordinates for both bounding boxes
        - Calculate intersection coordinates
        - IF no intersection:
            RETURN 0.0
        - Calculate intersection area
        - Calculate union area
        - RETURN IoU ratio
    
    METHOD prepare_data():
        - Get list of sequence folders
        - Split into training and test sets
        - Process training sequences with progress tracking
        - Process test sequences with progress tracking
        - Stack features and labels
        RETURN training and test data
    
    METHOD create_tracking_video(sequence_data, predictions, output_path):
        IF no sequence paths:
            RETURN
        
        - Read first image to get dimensions
        - Initialize video writer
        - Initialize motion tracker
        
        FOR each frame with predictions:
            - Load current image
            - Estimate camera motion for visualization
            
            IF not first frame:
                - Generate and draw search windows
                - Draw motion vectors on grid
            
            - Draw ground truth bounding box (green)
            - Draw predicted bounding box (red)
            - Calculate and display IoU
            - Display frame number
            - Write frame to video
        
        - Release video writer
    
    METHOD train_and_evaluate(output_dir):
        - Prepare training and test data
        - Scale features separately for position and size models
        - Split labels into position (x,y) and size (w,h)
        - Train Linear Regression for position prediction
        - Train Random Forest for size prediction
        - Create output directory
        
        FOR each test sequence:
            - Transform features using trained scalers
            - Predict positions and sizes separately
            - Combine predictions
            - Create tracking video
            - Store predictions and ground truth
        
        - Calculate evaluation metrics:
            * Position metrics (MAE, RMSE, R²) for x,y coordinates
            * Size metrics (MAE, RMSE, R²) for width,height
            * Overall IoU metric
        
        - Display comprehensive results
        RETURN evaluation metrics

MAIN FUNCTION:
    - Get dataset directory path from user
    - Set configuration parameters
    - Initialize tracking pipeline
    - Execute training and evaluation
    - Display summary results
    - Handle exceptions`,
        insights: 'A hybrid approach combining multiple feature descriptors (HOG for shape, SIFT for texture) provided a more robust tracking performance across varied lighting and object orientations than any single descriptor.'
      }
  },
];
