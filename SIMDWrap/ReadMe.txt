========================================================================
    			SIMDWrap Overview
========================================================================

SIMDWrap.cpp currently only contains tests. Include the file SIMD.h to get
started. Currently, only SIMD_SSE works. SIMD_AVX is not functional, and needs
reworking since AVX isn't a quarter of what AVX2 is.

The namespace is simd. So, building a new 4-element integer vector would be 
done with 

[cpp] SIMD4iv newVec();[/cpp]

/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h here only includes GLM stuff that you shouldn't need for the SIMD
headers.

/////////////////////////////////////////////////////////////////////////////
Other notes:

AppWizard uses "TODO:" comments to indicate parts of the source code you
should add to or customize.

/////////////////////////////////////////////////////////////////////////////
