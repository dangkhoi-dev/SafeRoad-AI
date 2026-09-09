@echo off
REM Day repo len GitHub (Windows). Chay tu thu muc goc cua repo.
REM
REM   scripts\push_to_github.bat
REM
setlocal
set REMOTE_URL=https://github.com/dangkhoi-dev/SafeRoad-AI.git

if not exist .git (
  echo Loi: khong thay thu muc .git. Hay chay tu thu muc goc SafeRoad-AI.
  exit /b 1
)

echo Repo   : %CD%
echo Remote : %REMOTE_URL%
echo.

git remote get-url origin >nul 2>&1
if %errorlevel%==0 (
  git remote set-url origin %REMOTE_URL%
  echo -^> Da cap nhat remote "origin"
) else (
  git remote add origin %REMOTE_URL%
  echo -^> Da them remote "origin"
)

echo.
echo Dang day len GitHub...
echo (Neu hoi mat khau, hay nhap Personal Access Token, KHONG phai mat khau tai khoan)
echo.

git push -u origin main
if %errorlevel% neq 0 (
  echo.
  echo Push that bai. Xem huong dan xu ly trong README hoac HUONG_DAN_TRIEN_KHAI.md
  exit /b %errorlevel%
)

echo.
echo Xong. Kiem tra tai: https://github.com/dangkhoi-dev/SafeRoad-AI
endlocal
