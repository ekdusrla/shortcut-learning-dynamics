$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$root = Split-Path $PSScriptRoot -Parent
$out = $PSScriptRoot
function Num($v) { if ([string]$v -eq 'nan') { return [double]::NaN }; [double]::Parse([string]$v, $culture) }
function Fmt($v) { ([double]$v).ToString('0.####', $culture) }
$runs = [ordered]@{}
$summary = @()
$segments = @()
$checks = @()
$inventory = @()
$testRows = @()
foreach ($pair in @(@('ResNet18','resnet18_corr095_seed42_e100'), @('VGG11_bn','vgg11_corr095_seed42_e100'))) {
    $name,$folder = $pair
    $dir = Join-Path $root "shortcut_results/runs/$folder"
    $rows = @(Import-Csv "$dir/history.csv")
    $runs[$name] = $rows
    Get-ChildItem $dir -Recurse -File | ForEach-Object { $inventory += [pscustomobject]@{model=$name;path=$_.FullName.Substring($root.Length+1);bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash} }
    $epochs = @($rows | ForEach-Object {[int]$_.epoch})
    if ($rows.Count -ne 101 -or @($epochs | Select-Object -Unique).Count -ne 101 -or @($epochs | Where-Object {$_ -lt 0 -or $_ -gt 100}).Count) { throw "Invalid epoch coverage: $name" }
    $groupCols = @($rows[0].PSObject.Properties.Name | Where-Object {$_ -match '^val_g\d\d_acc$'})
    foreach ($r in $rows) {
        $total=0; $correct=0.; $min=1.; $an=0; $cn=0; $ac=0.; $cc=0.
        foreach ($g in $groupCols) {
            $n=Num $r.($g -replace '_acc$','_n'); $a=Num $r.$g
            if ($n -le 0) {throw 'Empty group'}
            $total+=$n; $correct+=$n*$a; $min=[Math]::Min($min,$a)
            if ($g.Substring(5,1) -eq $g.Substring(6,1)) {$an+=$n;$ac+=$n*$a} else {$cn+=$n;$cc+=$n*$a}
        }
        if ($total -ne 5000 -or [Math]::Abs($correct/$total-(Num $r.val_acc)) -gt 1e-10 -or [Math]::Abs($min-(Num $r.val_wga)) -gt 1e-10 -or [Math]::Abs($ac/$an-(Num $r.val_aligned_acc)) -gt 1e-10 -or [Math]::Abs($cc/$cn-(Num $r.val_conflicting_acc)) -gt 1e-10) {throw "Metric consistency: $name epoch $($r.epoch)"}
        if ([Math]::Abs((Num $r.val_acc)-(Num $r.val_cf_acc)-(Num $r.val_cf_acc_drop)) -gt 1e-10) {throw 'CF drop mismatch'}
        $r | Add-Member -NotePropertyName aligned_conflict_gap -NotePropertyValue ((Num $r.val_aligned_acc)-(Num $r.val_conflicting_acc))
    }
    $rows | Export-Csv "$out/${name}_epochs.csv" -NoTypeInformation -Encoding UTF8
    $test = Get-Content "$dir/test_metrics.json" -Raw | ConvertFrom-Json
    foreach ($p in $test.metrics.PSObject.Properties) { $testRows += [pscustomobject]@{model=$name;epoch=$test.epoch;metric=$p.Name;value=$p.Value} }
    $checks += [pscustomobject]@{model=$name;rows=$rows.Count;groups=$groupCols.Count;validation_samples=$total;aligned_samples=$an;conflict_samples=$cn;latest_epoch=(Get-Content "$dir/latest.json" -Raw|ConvertFrom-Json).epoch;test_epoch=$test.epoch;metric_consistency='PASS';lr_values=($rows.lr|Select-Object -Unique)-join ','}
    foreach ($metric in @('train_acc','val_acc','val_wga','val_aligned_acc','val_conflicting_acc','val_cf_flip_rate','val_cf_acc','val_cf_acc_drop','aligned_conflict_gap')) {
        $vals=@($rows|Where-Object {[int]$_.epoch -ge 1}|ForEach-Object {Num $_.$metric})
        $late=@($rows|Where-Object {[int]$_.epoch -ge 91}|ForEach-Object {Num $_.$metric})
        $inc=0;$dec=0
        for($i=1;$i -lt $vals.Count;$i++){if($vals[$i]-gt $vals[$i-1]+1e-12){$inc++};if($vals[$i]-lt $vals[$i-1]-1e-12){$dec++}}
        $summary += [pscustomobject]@{model=$name;metric=$metric;epoch0=Num $rows[0].$metric;epoch1=Num $rows[1].$metric;epoch100=Num $rows[100].$metric;delta_1_100=(Num $rows[100].$metric)-(Num $rows[1].$metric);mean_91_100=($late|Measure-Object -Average).Average;min_1_100=($vals|Measure-Object -Minimum).Minimum;max_1_100=($vals|Measure-Object -Maximum).Maximum;increases=$inc;decreases=$dec}
        foreach($range in @(@(1,10),@(11,30),@(31,60),@(61,100))) {
            $sub=@($rows|Where-Object {[int]$_.epoch -ge $range[0] -and [int]$_.epoch -le $range[1]})
            $xm=($sub|ForEach-Object {[double]$_.epoch}|Measure-Object -Average).Average
            $ym=($sub|ForEach-Object {Num $_.$metric}|Measure-Object -Average).Average
            $xy=0.;$xx=0.;foreach($r in $sub){$dx=[double]$r.epoch-$xm;$xy+=$dx*((Num $r.$metric)-$ym);$xx+=$dx*$dx}
            $segments += [pscustomobject]@{model=$name;metric=$metric;start=$range[0];end=$range[1];slope_pp_per_epoch=100*$xy/$xx}
        }
    }
}
$inventory|Export-Csv "$out/file_inventory.csv" -NoTypeInformation -Encoding UTF8
$testRows|Export-Csv "$out/final_test_metrics.csv" -NoTypeInformation -Encoding UTF8
$checks|Export-Csv "$out/validation_checks.csv" -NoTypeInformation -Encoding UTF8
$summary|Export-Csv "$out/summary.csv" -NoTypeInformation -Encoding UTF8
$segments|Export-Csv "$out/segment_slopes.csv" -NoTypeInformation -Encoding UTF8
function Plot($filename,$title,$series,$xmin=0,$ymin=0,$ymax=100) {
    $colors=@('#2563eb','#dc2626','#059669','#9333ea','#d97706','#0891b2','#64748b')
    $svg=[Collections.Generic.List[string]]::new()
    $svg.Add('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560"><rect width="1000" height="560" fill="white"/><g font-family="Arial,sans-serif" font-size="13" fill="#172033">')
    $svg.Add("<text x='70' y='28' font-size='20'>$title</text>")
    for($t=0;$t -le 5;$t++){$y=440-$t*72;$v=$ymin+($ymax-$ymin)*$t/5;$svg.Add("<line x1='70' y1='$y' x2='960' y2='$y' stroke='#e2e8f0'/><text x='12' y='$($y+4)'>$(Fmt $v)%</text>")}
    for($e=$xmin;$e -le 100;$e+=10){$x=70+890*($e-$xmin)/(100-$xmin);$svg.Add("<text x='$(Fmt $x)' y='465'>$e</text>")}
    $j=0
    foreach($s in $series){$points=@();foreach($r in $runs[$s.model]){if([int]$r.epoch -lt $xmin){continue};$v=Num $r.($s.metric);if([double]::IsNaN($v)){continue};$x=70+890*([int]$r.epoch-$xmin)/(100-$xmin);$y=440-360*(100*$v-$ymin)/($ymax-$ymin);$points+="$(Fmt $x),$(Fmt $y)"};$c=$colors[$j%$colors.Count];$svg.Add("<polyline points='$($points -join ' ')' fill='none' stroke='$c' stroke-width='1.8'/>");$lx=70+($j%3)*300;$ly=505+[Math]::Floor($j/3)*20;$svg.Add("<text x='$lx' y='$ly' fill='$c'>$($s.label)</text>");$j++}
    $svg.Add('<text x="470" y="485">Epoch</text></g></svg>')
    [IO.File]::WriteAllLines("$out/$filename.svg",$svg,[Text.UTF8Encoding]::new($false))
}
foreach($name in $runs.Keys){
    $s=@();foreach($metric in @('train_acc','val_acc','val_wga','val_aligned_acc','val_conflicting_acc','val_cf_acc')){$s+=@{model=$name;metric=$metric;label=$metric}}
    Plot "${name}_accuracy" "$name accuracy (epoch 0-100)" $s
    Plot "${name}_accuracy_zoom" "$name accuracy (epoch 1-100, zoom)" $s 1 55 100
    Plot "${name}_shortcut" "$name shortcut sensitivity (epoch 0-100)" @(@{model=$name;metric='val_cf_flip_rate';label='Flip rate'},@{model=$name;metric='aligned_conflict_gap';label='Aligned - conflict'},@{model=$name;metric='val_cf_acc_drop';label='CF accuracy drop'}) 0 -1 8
}
foreach($metric in @('train_acc','val_acc','val_wga','val_aligned_acc','val_conflicting_acc','val_cf_flip_rate','val_cf_acc','val_cf_acc_drop','aligned_conflict_gap')) {
    $s=@();foreach($name in $runs.Keys){$s+=@{model=$name;metric=$metric;label=$name}}
    $lo=0;$hi=100;if($metric -in @('val_cf_flip_rate','aligned_conflict_gap')){$lo=-1;$hi=8};if($metric -eq 'val_cf_acc_drop'){$lo=-1;$hi=1}
    Plot "compare_$metric" "$metric : ResNet18 vs VGG11_bn (epoch 0-100)" $s 0 $lo $hi
}
$html='<meta charset="utf-8"><title>Shortcut Learning Dynamics</title><style>body{font:16px Arial;max-width:1100px;margin:32px auto}img{width:100%}</style><h1>Shortcut Learning Dynamics</h1><p>Observed metrics only. Accuracy values in percent. Epoch 0 train accuracy is missing. See report_ko.md for interpretation.</p>'
Get-ChildItem $out -Filter '*.svg'|ForEach-Object {$html+="<h2>$($_.BaseName)</h2><img src='$($_.Name)'/>"}
[IO.File]::WriteAllText("$out/graphs.html",$html,[Text.UTF8Encoding]::new($false))
$checks | Format-Table
$summary | Where-Object {$_.metric -in 'val_cf_flip_rate','val_wga','val_conflicting_acc'} | Format-Table
$segments | Where-Object {$_.metric -eq 'val_cf_flip_rate'} | Format-Table
