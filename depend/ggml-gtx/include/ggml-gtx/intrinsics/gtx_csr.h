//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX architecture CSR definition
// Author      : mh.kim ( NPU Div - NPU Core Team )
// Last Update : 2025/12/15
//==================================================================

#ifndef GTX_CSR_H
#define GTX_CSR_H

#ifdef __cplusplus
extern "C" {
#endif


//=================================
// Global (GSPR, GTX) / 000-3FF (Nest관련)
//=================================
// Run
#define RUN                 0x000

// Command
#define OPERAND_1           0x001
#define OPERAND_2           0x002
#define OPERAND_3           0x003
#define OPCODE              0x004
// [추가] Nest Selection (공유 윈도우 제어의 핵심)
#define NEST_SELECT         0x005  // 현재 CPU 주소창에 연결할 Nest ID (0-3) 선택
#define NEST_STATUS         0x382  // 각 Nest의 전원/연결 상태 확인 (is_nx_accessible용)
// Stack
#define STACK_INFO          0x010
#define STACK_SAVE          0x011

// Performance - reserved

// H/W Feature & Debug
#define COMMAND             0x300
#define INTERRUPT_CONTROL   0x310
#define INTERRUPT_VECTOR    0x311
#define INTERRUPT_SOURCE    0x312
#define SPU_BUSY            0x380
#define SMU_BUSY            0x381
#define INFO                0x3F0
#define ID                  0x3F1


//=================================
// Nest (NSPR, NEST) / 400-7FF (SMU, THREAD관련)
//=================================
// Mask
#define THREAD              0x400
#define SHARED              0x401

// Config
#define DATA_FORMAT         0x402
#define OP_MODE             0x403

// Performance - reserved

// H/W Feature & Debug
#define CLEAR               0x700
#define NEST_CREDIT         0x701
#define SDLE_STATUS         0x780
#define CREDIT_COUNT        0x781
#define CREDIT_ERROR        0x782


//=================================
// Local (LSPR, SPU) / 800-BFF (SPU관련)
//=================================
// SCR
#define SGPR_0              0x800
#define SGPR_1              0x801
#define SGPR_2              0x802
#define SGPR_3              0x803
#define SGPR_4              0x804
#define SGPR_5              0x805
#define SGPR_6              0x806
#define SGPR_7              0x807
#define SGPR_8              0x808
#define SGPR_9              0x809
#define SGPR_10             0x80a
#define SGPR_11             0x80b
#define SGPR_12             0x80c
#define SGPR_13             0x80d
#define SGPR_14             0x80e
#define SGPR_15             0x80f

#define SGPR_16             0x810
#define SGPR_17             0x811
#define SGPR_18             0x812
#define SGPR_19             0x813
#define SGPR_20             0x814
#define SGPR_21             0x815
#define SGPR_22             0x816
#define SGPR_23             0x817
#define SGPR_24             0x818
#define SGPR_25             0x819
#define SGPR_26             0x81a
#define SGPR_27             0x81b
#define SGPR_28             0x81c
#define SGPR_29             0x81d
#define SGPR_30             0x81e
#define SGPR_31             0x81f

#define SGPR_32             0x820
#define SGPR_33             0x821
#define SGPR_34             0x822
#define SGPR_35             0x823
#define SGPR_36             0x824
#define SGPR_37             0x825
#define SGPR_38             0x826
#define SGPR_39             0x827
#define SGPR_40             0x828
#define SGPR_41             0x829
#define SGPR_42             0x82a
#define SGPR_43             0x82b
#define SGPR_44             0x82c
#define SGPR_45             0x82d
#define SGPR_46             0x82e
#define SGPR_47             0x82f

#define SGPR_48             0x830
#define SGPR_49             0x831
#define SGPR_50             0x832
#define SGPR_51             0x833
#define SGPR_52             0x834
#define SGPR_53             0x835
#define SGPR_54             0x836
#define SGPR_55             0x837
#define SGPR_56             0x838
#define SGPR_57             0x839
#define SGPR_58             0x83a
#define SGPR_59             0x83b
#define SGPR_60             0x83c
#define SGPR_61             0x83d
#define SGPR_62             0x83e
#define SGPR_63             0x83f

#define SGPR_64             0x840
#define SGPR_65             0x841
#define SGPR_66             0x842
#define SGPR_67             0x843
#define SGPR_68             0x844
#define SGPR_69             0x845
#define SGPR_70             0x846
#define SGPR_71             0x847
#define SGPR_72             0x848
#define SGPR_73             0x849
#define SGPR_74             0x84a
#define SGPR_75             0x84b
#define SGPR_76             0x84c
#define SGPR_77             0x84d
#define SGPR_78             0x84e
#define SGPR_79             0x84f

#define SGPR_80             0x850
#define SGPR_81             0x851
#define SGPR_82             0x852
#define SGPR_83             0x853
#define SGPR_84             0x854
#define SGPR_85             0x855
#define SGPR_86             0x856
#define SGPR_87             0x857
#define SGPR_88             0x858
#define SGPR_89             0x859
#define SGPR_90             0x85a
#define SGPR_91             0x85b
#define SGPR_92             0x85c
#define SGPR_93             0x85d
#define SGPR_94             0x85e
#define SGPR_95             0x85f

#define SGPR_96             0x860
#define SGPR_97             0x861
#define SGPR_98             0x862
#define SGPR_99             0x863
#define SGPR_100            0x864
#define SGPR_101            0x865
#define SGPR_102            0x866
#define SGPR_103            0x867
#define SGPR_104            0x868
#define SGPR_105            0x869
#define SGPR_106            0x86a
#define SGPR_107            0x86b
#define SGPR_108            0x86c
#define SGPR_109            0x86d
#define SGPR_110            0x86e
#define SGPR_111            0x86f

#define SGPR_112            0x870
#define SGPR_113            0x871
#define SGPR_114            0x872
#define SGPR_115            0x873
#define SGPR_116            0x874
#define SGPR_117            0x875
#define SGPR_118            0x876
#define SGPR_119            0x877
#define SGPR_120            0x878
#define SGPR_121            0x879
#define SGPR_122            0x87a
#define SGPR_123            0x87b
#define SGPR_124            0x87c
#define SGPR_125            0x87d
#define SGPR_126            0x87e
#define SGPR_127            0x87f

#define SPM_ADDR_A          0x900
#define SPM_ADDR_B          0x901
#define SPM_ADDR_C          0x902
#define SPM_ADDR_R          0x903

// Performance - reserved

// H/W Feature & Debug
#define CP_CONTROL          0xB00
#define DMA_CONTROL         0xB20
// [추가] DMA 상세 제어 (DDR -> L2 -> L1 복사 시 필수)
#define DMA_SRC_ADDR_LO     0xB21  // DMA 소스 주소 (DDR) 하위 32비트
#define DMA_SRC_ADDR_HI     0xB22  // DMA 소스 주소 (DDR) 상위 32비트
#define DMA_DST_ADDR        0xB23  // DMA 목적지 주소 (L2/L1 Window 주소)
#define DMA_TRANS_SIZE      0xB24  // 전송 크기 (Byte 단위)
// [추가] 동기화 (복사가 끝났는지 확인)
#define DMA_DONE_INTR       0xBA1  // DMA 완료 인터럽트 상태 및 클리어

#define MXE_CONTROL         0xB30
#define SDE_CONTROL         0xB40
#define PDE_CONTROL         0xB50
#define SVR_TIMING          0xB60
#define SPU_STATUS          0xB80
#define DMA_STATUS          0xBA0
#define MXE_STATUS          0xBB0
#define SDE_STATUS          0xBC0
#define PDE_STATUS          0xBD0
#define CREDIT              0xBE0
#define THREAD_ID           0xBFF


//=================================
// System / C00-FFF
//=================================
// reserved

#ifdef __cplusplus
}
#endif

#endif