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
    link: 'https://github.com/ayushsaun24024',
    image: 'https://placehold.co/600x400.png',
    aiHint: 'audio waveform',
    details: {
      flowchart: `graph TD
    A[Ingest Audio Data] --> B(Pre-process)
    B --> C{Extract Features}
    C --> D[Input to SSL Encoder]
    D --> E(Fine-tune with CNN Head)
    E --> F((Calculate EER))`,
      pseudocode: `function train_model(data):
  model = load_ssl_encoder()
  model.add_head(cnn_classifier)
  optimizer = Adam(model.parameters)
  
  for epoch in epochs:
    for batch in data_loader(data):
      audio, label = batch
      output = model(audio)
      loss = cross_entropy(output, label)
      loss.backward()
      optimizer.step()
  return model`,
      insights: 'The key insight was that Self-Supervised Learning (SSL) models pre-trained on vast amounts of unlabeled speech data could capture subtle, low-level features indicative of spoofing artifacts that traditional methods miss.'
    }
  },
  {
    title: 'Automatic Speaker Verification System',
    description: 'Built an end-to-end biometric speaker-verification pipeline using deep speaker embeddings and ensemble score fusion. Trained on 350h of multilingual speech, trimming tandem EER to 30%.',
    technologies: ['Python', 'PyTorch', 'SpeechBrain', 'HuggingFace Transformers', 'Docker', 'Linux'],
    link: 'https://github.com/ayushsaun24024',
    image: 'https://placehold.co/600x400.png',
    aiHint: 'voice recognition',
    details: {
        flowchart: `graph TD
    A[Load VoxCeleb Dataset] --> B(Extract Speaker Embeddings)
    B --> C{Implement Scoring Backend}
    C --> D(Calibrate Scores)
    D --> E[Fuse Scores]
    E --> F((Evaluate EER))`,
        pseudocode: `function verify_speaker(audio1, audio2):
  embedding1 = model.extract_embedding(audio1)
  embedding2 = model.extract_embedding(audio2)
  
  score = cosine_similarity(embedding1, embedding2)
  
  if score > threshold:
    return "Same Speaker"
  else:
    return "Different Speakers"`,
        insights: 'Ensemble methods and score calibration were critical for robustness. Fusing scores from different embedding models significantly reduced the Equal Error Rate compared to any single model.'
      }
  },
  {
    title: 'Classic ML-Based Vocoder',
    description: 'Engineered a Mel-spectrogram-to-waveform vocoder using STFT and Griffin-Lim, delivering studio-grade audio with a Mean Opinion Score (MOS) of 4.1. Streamlined TensorFlow inference for 30% lower latency.',
    technologies: ['Python', 'TensorFlow', 'Librosa', 'STFT', 'Griffin-Lim', 'Docker'],
    image: 'https://placehold.co/600x400.png',
    aiHint: 'sound engineering',
    details: {
        flowchart: `graph TD
    A[Generate Mel-Spectrogram] --> B{Initialize Phase}
    B --> C(Griffin-Lim Loop)
    C --> D{Synthesize Waveform}
    D --> E{Re-estimate Phase}
    E --> F{Replace Magnitude}
    F --> C
    C --> G((Converged Waveform))`,
        pseudocode: `function griffin_lim(mel_spec, iterations):
  phase = random_phase()
  for i in 0..iterations:
    waveform = inverse_stft(mel_spec, phase)
    stft_matrix = stft(waveform)
    phase = get_phase(stft_matrix)
  
  return waveform`,
        insights: 'The Griffin-Lim algorithm is an iterative process that can reconstruct a time-domain signal from only the magnitude of its Short-Time Fourier Transform (STFT), making it a powerful tool for vocoding tasks when phase information is lost.'
      }
  },
  {
    title: 'Single-Object Tracking System',
    description: 'Delivered a real-time 30 FPS tracker with camera-motion compensation. Integrated hybrid descriptors and ensemble regressors to achieve 85% IoU and 20% lower MAE.',
    technologies: ['Python', 'OpenCV', 'HOG', 'SIFT', 'Random Forest', 'Linear Regression'],
    link: 'https://github.com/ayushsaun24024',
    image: 'https://placehold.co/600x400.png',
    aiHint: 'object tracking',
    details: {
        flowchart: `graph TD
    A[Initialize Tracker] --> B{For each frame}
    B --> C(Extract Features)
    C --> D[Predict Location]
    D --> E{Apply Motion Compensation}
    E --> F[Update Bounding Box]
    F --> B`,
        pseudocode: `function track_object(video_frames):
  tracker = initialize_tracker(first_frame)
  
  for frame in video_frames:
    features = extract_features(frame, tracker.bbox)
    new_bbox = tracker.predict(features)
    tracker.update(new_bbox)
    draw_bbox(frame, new_bbox)
  
  return tracked_video`,
        insights: 'A hybrid approach combining multiple feature descriptors (HOG for shape, SIFT for texture) provided a more robust tracking performance across varied lighting and object orientations than any single descriptor.'
      }
  },
];
