# 화재 및 외벽 오염 감지 시스템 개발 로드맵 (수정판)

## 📊 프로젝트 개요

### 🎯 정량 목표
| 구분 | 최소 요구사항 | 목표 | 측정 방법 |
|------|---------------|------|-----------|
| **외벽 오염 인식** | mAP@0.5 ≥ 70% | mAP@0.5 ≥ 75% | COCO Evaluation |
| **화재/연기 감지** | Recall ≥ 80% | Recall ≥ 85% | Binary Classification |
| **추론 속도** | Jetson Nano ≥ 15 FPS | Jetson Orin ≥ 30 FPS | TensorRT 측정 |
| **시스템 성능** | CCTV 연동 성공률 ≥ 90% | 리포트 자동생성 100% | 통합 테스트|

### ⚙️ 기술 스택
| 카테고리 | 기술/도구 | 버전/설정 |
|----------|-----------|-----------|
| **백본 모델** | YOLOv10n | Ultralytics 최신 |
| **개발 프레임워크** | PyTorch | 2.0+ (MPS/CUDA) |
| **효율적 학습** | LoRA, Adapter | PEFT Library |
| **배포 최적화** | TensorRT | 8.0+ (경량화는 추후) |

### 🛠 환경별 개발 스택
| 환경 | 기간 | 주요 스택 | 용도 |
|------|------|-----------|------|
| **Mac M3 Max (로컬)** | 9월 | PyTorch MPS, Jupyter | 초기 프로토타입 |
| **DPG 클라우드** | 10월~ | PyTorch CUDA, Multi-GPU | 본격 개발 |
| **Jetson Edge** | 최종 배포 | TensorRT, DeepStream | 실시간 추론 |

### 📊 데이터 전략 (오픈소스 활용)
| 도메인 | Youtube 수집집 | 자체 수집 | 총 데이터 |
|--------|-------------------|-----------|-----------|
| **화재/연기** | Fire, Smoke Detection (2,500장) | 교육용 500장 | 3,000장 |
| **외벽 오염** | Crack Detection (2,500장) | 교육용 500장 | 3,000장 |

### 🔧 라벨링 전략 (교육용 중심)
| 목적 | 데이터 양 | 작업자 | 도구 | 소요 기간 |
|------|-----------|--------|------|----------|
| **직원 교육** | 500장 × 2도메인 | 내부 AI팀 | CVAT 등 (무료) | 1주 |
| **품질 기준** | 교육용 라벨링 결과 | AI팀 | 검증 스크립트 | 2일 |
| **오픈소스 검증** | 기존 라벨 검토 | AI팀 | 자동 변환 | 3일 |

---

## ☁️ DPG 클라우드 환경 구축 가이드

### 클라우드 스펙 요구사항
| 리소스 | 최소 사양 | 권장 사양 | 용도 |
|--------|-----------|-----------|------|
| **GPU** | RTX 4090 1장 | RTX 5090 2장 | 모델 학습 |
| **CPU** | 16 Core | 32 Core | 데이터 처리 |
| **Memory** | 64GB | 128GB | 대용량 배치 |
| **Storage** | 1TB SSD | 2TB NVMe | 데이터셋 저장 |

### 클라우드 환경 셋업
| 단계 | 작업 | 설치 항목 | 설정 |
|------|------|-----------|------|
| **Base** | OS 환경 | Ubuntu 22.04 LTS | CUDA 12.0+ |
| **Python** | 가상환경 | conda, pip | Python 3.9+ |
| **Deep Learning** | 프레임워크 | PyTorch, CUDA | Multi-GPU 설정 |
| **Development** | 개발 도구 | Jupyter, VSCode Server | 원격 접속 |
| **MLOps** | 실험 관리 | Weights & Biases, DVC | 버전 관리 |

### 클라우드 설치 스크립트
```bash
# DPG 클라우드 환경 구축 스크립트
#!/bin/bash

# 1. CUDA 설치
sudo apt update
sudo apt install nvidia-driver-535 cuda-toolkit-12-0

# 2. Docker 설치 (옵션)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Conda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 4. 프로젝트 환경 생성
conda create -n fire_detection python=3.9
conda activate fire_detection

# 5. PyTorch + CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. YOLOv10 및 관련 라이브러리
pip install ultralytics peft transformers
pip install opencv-python albumentations wandb
pip install tensorrt onnx onnxruntime-gpu
```

---

## 📊 개발 전략

### Parameter-Efficient Fine-Tuning (경량화 기법)
| 기법 | 분류 | 효과 | 적용 시점 | 우선순위 |
|------|------|------|----------|----------|
| **HEAD 튜닝** | 기본 방법 | 파라미터 90% 절약 | 첫 번째 | 1순위 |
| **LoRA** | 경량화 기법 | 파라미터 95% 절약 | 성능 부족 시 | 2순위 |
| **Adapter** | 경량화 기법 | 파라미터 85% 절약 | 대안 방법 | 3순위 |

### 기존 경량화 기법 (추후 적용)
| 기법 | 효과 | 적용 조건 | 예상 기간 |
|------|------|----------|----------|
| **Pruning** | 모델 크기 50% 감소 | 시간 여유 시 | +2주 |
| **Quantization** | 메모리 75% 절약 | 극한 최적화 필요 시 | +1주 |

---

## 🔥 Phase 1: 화재/연기 감지 모델 개발 (3개월)

### Month 1: 데이터 구축 (9월 - Mac M3)
| Week | 작업 | 세부 내용 | 산출물 |
|------|------|-----------|---------|
| **W1** | 직접 수집 | Fire, Smoke Youtube & 직접 수집 | 2,500장 Dataset |
| **W1** | 데이터 포맷 통일 | YOLO format 변환, 클래스 매핑 | 표준화된 Dataset |
| **W2** | 교육용 라벨링 | 직원 500장 라벨링 교육 및 수행 | 교육 Dataset + 숙련도 |
| **W2** | 데이터 검증 | 유튜브 수집 + 교육용 품질 통합 검증 | 검증된 3,000장 |
| **W3** | Train/Val/Test 분할 | 7:2:1 비율 분할 | 학습 준비 Dataset |
| **W4** | Baseline 구축 | YOLOv10n HEAD 튜닝 | 화재 Baseline Model |

### Month 2: 모델 최적화 (10월 - DPG 클라우드)
| Week | 작업 | 세부 내용 | 산출물 |
|------|------|-----------|---------|
| **W1** | 클라우드 마이그레이션 | 환경 구축 + 데이터 이전 | DPG 개발 환경 |
| **W1** | 성능 분석 | Baseline 정밀 성능 측정 | 성능 분석 리포트 |
| **W2** | LoRA 적용 | HEAD + Neck 선택적 LoRA | LoRA 화재 모델 |
| **W2** | Adapter 실험 | Multi-layer Adapter 적용 | Adapter 화재 모델 |
| **W3** | 기법 비교 | HEAD/LoRA/Adapter 성능 비교 | 최적 기법 선정 |
| **W4** | 최종 최적화 | 선정 기법 하이퍼파라미터 튜닝 | 최적화된 화재 모델 |

### Month 3: 배포 최적화 (11월 - DPG 클라우드)
| Week | 작업 | 세부 내용 | 산출물 |
|------|------|-----------|---------|
| **W1** | TensorRT 변환 | PyTorch → ONNX → TensorRT | TRT Engine |
| **W2** | Jetson 성능 테스트 | Nano/Orin FPS 실측 | Jetson 성능 리포트 |
| **W3** | CCTV 연동 개발 | RTSP 실시간 스트림 처리 | CCTV 연동 모듈 |
| **W4** | 화재 시스템 완성 | 통합 테스트 및 문서화 | 완성된 화재 시스템 |

---

## 🏗️ Phase 2: 외벽 오염 감지 모델 개발 (3개월)

### Month 4: 외벽 데이터 및 Transfer Learning (12월)
| Week | 작업 | 세부 내용 | 산출물 |
|------|------|-----------|---------|
| **W1** | 외벽 오픈소스 수집 | Crack Detection, Concrete Damage 데이터셋 | 2,500장 외벽 Dataset |
| **W1** | 외벽 교육 라벨링 | Crack/Stain/Rust 500장 교육 작업 | 외벽 교육 Dataset |
| **W2** | Transfer Learning | 화재 모델 구조 → 외벽 적용 | 외벽 Transfer 모델 |
| **W3** | 도메인 적응 | 외벽 특화 증강 및 Loss 튜닝 | 적응된 외벽 모델 |
| **W4** | 성능 검증 | mAP@0.5 70% 목표 달성 확인 | 외벽 Baseline |

### Month 5: 외벽 모델 최적화 (1월)
| Week | 작업 | 세부 내용 | 산출물 |
|------|------|-----------|---------|
| **W1** | PEFT 적용 | 화재 모델 경험 활용 LoRA/Adapter | 최적화된 외벽 모델 |
| **W2** | 클래스 특화 | Crack 감지 정확도 집중 향상 | 고성능 외벽 모델 |
| **W3** | mAP 75% 달성 | 목표 성능 달성을 위한 최종 튜닝 | 목표 달성 모델 |
| **W4** | 배포 준비 | 외벽 모델 TensorRT 변환 | 배포 가능 외벽 모델 |

### Month 6: 통합 시스템 완성 (2월)
| Week | 작업 | 세부 내용 | 산출물 |
|------|------|-----------|---------|
| **W1** | 듀얼 모델 통합 | 화재+외벽 동시 실행 최적화 | 통합 추론 시스템 |
| **W2** | 시스템 통합 | CCTV + 실시간 분석 + 알림 | 완전한 파이프라인 |
| **W3** | 관리자 시스템 | 리포트 자동화 + 대시보드 | 관리자 인터페이스 |
| **W4** | 최종 검수 | 전체 정량 목표 달성 검증 | 최종 시스템 |

---

## 🗂️ 추천 오픈소스 데이터셋

### 외벽 오염/손상 감지  
| 데이터셋명 | 이미지 수 | 클래스 | 라이선스 | 품질 |
|-----------|-----------|-------|----------|-------|
| **Crack Detection** | 4,000+ | Crack, No-Crack | Academic | 높음 |
| **Concrete Damage** | 2,800+ | Crack, Stain, Normal | MIT | 중간 |
| **Building Defect** | 3,500+ | Crack, Rust, Paint | Open | 중간 |
| **Infrastructure** | 2,200+ | Crack, Corrosion, Clean | CC BY | 높음 |

---

## 📈 예상 성과 및 효율성

### 기간 단축 효과
| 구분 | 기존 계획 | 수정된 계획 | 단축 효과 |
|------|-----------|-------------|-----------|
| **총 개발 기간** | 8개월 | 6개월 | 25% 단축 |

### Parameter-Efficient 기법 효과
| 기법 | 학습 시간 단축 | 메모리 절약 | 성능 유지 | 적용 난이도 |
|------|---------------|-------------|-----------|-------------|
| **HEAD 튜닝** | 70% | 80% | 95% | 쉬움 |
| **LoRA** | 80% | 90% | 98% | 보통 |
| **Adapter** | 60% | 75% | 96% | 쉬움 |

### 최종 목표 달성 확률
- **화재 감지 Recall 85%**: 95% 확률
- **외벽 인식 mAP 75%**: 90% 확률  
- **Jetson 15 FPS**: 98% 확률
- **전체 시스템 통합**: 90% 확률