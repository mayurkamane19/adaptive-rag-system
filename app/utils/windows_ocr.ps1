param(
    [Parameter(Mandatory=$true)]
    [string]$ImagePath
)

try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime -ErrorAction Stop
    $asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }
    Function Await($WinRtTask, $ResultType) {
        $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
        $netTask = $asTask.Invoke($null, @($WinRtTask))
        $netTask.Wait(-1) | Out-Null
        $netTask.Result
    }

    [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrEngine, Windows.Foundation.UniversalApiContract, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null

    $resolvedPath = (Resolve-Path -LiteralPath $ImagePath).Path
    $fileTask = [Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPath)
    $storageFile = Await $fileTask ([Windows.Storage.StorageFile])
    $streamTask = $storageFile.OpenAsync([Windows.Storage.FileAccessMode]::Read)
    $stream = Await $streamTask ([Windows.Storage.Streams.IRandomAccessStream])
    $decoderTask = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)
    $decoder = Await $decoderTask ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmapTask = $decoder.GetSoftwareBitmapAsync()
    $bitmap = Await $bitmapTask ([Windows.Graphics.Imaging.SoftwareBitmap])

    $lang = New-Object Windows.Globalization.Language("en-US")
    $ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
    if (-not $ocrEngine) {
        $ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    }
    if (-not $ocrEngine) {
        exit 2
    }
    $ocrTask = $ocrEngine.RecognizeAsync($bitmap)
    $ocrResult = Await $ocrTask ([Windows.Media.Ocr.OcrResult])

    if ($ocrResult -and $ocrResult.Lines) {
        $textOut = ($ocrResult.Lines | ForEach-Object { $_.Text }) -join "`n"
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        Write-Output $textOut
    }
} catch {
    Write-Error $_
    exit 1
}
